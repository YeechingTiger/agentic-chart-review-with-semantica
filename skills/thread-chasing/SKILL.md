---
name: thread-chasing
description: Use when a document defers its own conclusion or a read comes back incomplete - a report says pending, stains pending, preliminary, deferred, to be dictated, see addendum, amended, additional sections, correlate clinically, or names an outside facility; a read_document result has truncated true; a search hits a report you have only partly read; or you are about to record a value taken from an interim line. Tells you what each marker obliges, the three places the resolving text lives and the order to try them, and what to record when the thread cannot be closed.
slot: general
license: MIT
---

# Chasing an unsettled thread to where it was settled

A chart is written forward in time by people who did not yet know the answer. Any document
may defer its own conclusion. **A deferred conclusion is not evidence for the conclusion —
it is an instruction about where to look next.** Applies to any variable.

## The failure this exists to prevent

On 2026-07-26, patient `P05` was coded `8046` (non-small cell carcinoma, NOS)
from a line saying special stains were pending. The document that resolved those stains was
in the same chart, in the same file. Measured from the files themselves:

| | |
|---|---|
| the two `Surgical-Pathology-Document`s | 19,050 and 20,199 characters |
| `special stain` first occurs at offset | 899 and 831 — **inside** a default read |
| `addendum` first occurs at offset | 4,353 and 4,729 — **outside** it |
| `pending` first occurs at offset | 14,708 and 15,630 — far outside it |
| `read_document` default `limit` | 4,000 |

The run read the first 4,000 characters, saw the unsettled line, and stopped 353 characters
short of the word that resolved it. It then spent its entire 400k-token budget defending the
interim answer. It was not short of information; it was short of one more call.

This is systematic, not a one-off. Across the 39 traces in `runs/`, **112 `read_document`
results came back `truncated: true` and only 11 (10%) were ever continued.**

## Three places the resolution lives, in the order to try them

1. **The rest of the same document.** Cheapest and most often correct — in the case above
   both the addendum and the final wording were in the same file. Page with
   `read_document(note_id, offset=<previous offset + returned_chars>)` until `truncated` is
   false. Treat `truncated: true` as an open thread in its own right, whether or not you
   saw a marker word.
2. **A named section of it.** Call `read_section(note_id, "")` with an empty section to get
   the heading list, then read by exact name. Do not guess the name: on these charts the
   real headings are `ADDENDUM 1` and `ADDENDUM 2`, so an exact-match lookup for `ADDENDUM`
   returns nothing and looks like absence. Many reports carry no headings at all; when the
   list comes back empty, fall back to paging.
3. **A later document.** `Path-Rpt-Addendum`, `Addendum` and
   `Discharge-Summary-Staff-Addendum` are their own document types in this corpus's
   1,516-type vocabulary — an addendum is often a separate note with its own date. Use
   `list_documents(date_from=<the deferring report's date>)` or `timeline`, and look at
   the same and neighbouring types.

## What each marker obliges

| marker | what it is telling you | discharge it by |
|---|---|---|
| `pending`, `results pending`, `stains pending`, `additional sections`, `recut` | the lab had not finished when this text was signed | finish the document, then look for a later report of the same type |
| `see addendum`, `addendum`, `amended` | a resolution exists and is filed somewhere | section list first, then the addendum document types |
| `see synoptic`, `synoptic report` | the narrative diagnosis is a **summary**; the CAP-protocol block behind it carries the subtype, grade and stage the narrative left out | the same file, further down — read to the end, do not stop at the narrative diagnosis |
| `preliminary` | a final version will follow | check the type before spending a step — in a 50-patient sample `preliminary` sat mostly in `Blood-Culture` and imaging, rarely in pathology |
| `deferred`, `in consultation`, `sent out` | another pathologist or lab holds it | later documents; may legitimately never return |
| `correlate clinically`, `clinical correlation` | the author is saying **their own document cannot settle it** | a *different class* of document, not a later version of this one |
| `to be dictated` | the text does not exist yet | a later document only; nothing to page to |
| `outside facility`, `outside hospital`, `outside institution` | the document is not in this chart | **you cannot discharge this by reading.** See below |

Base rates and the false-positive traps for each marker are in
`skills/thread-chasing/references/marker-catalogue.md` — consult it before spending several
steps on a marker, because most instances of `pending` and `deferred` in this corpus are
ordinary clinic chatter, not a lab thread.

## When the thread cannot be closed

An unclosed thread is a finding, and it must leave a trace:

- Record the marker itself with `record_evidence(..., stance="contradicts")`. A pending line
  cited as contradicting evidence is the difference between "the chart is unsettled" and
  "the reviewer did not notice".
- State in your reasoning **which of the three places you looked**, not merely that you
  looked.
- Never treat a mention that a test was performed elsewhere as evidence of what it showed.
  A referral letter naming an outside biopsy tells you a biopsy happened, and nothing about
  its result.
- Then answer the abstention status the spec names, for the fields the thread bore on only.
  A pending stain blocks the tissue field; it usually does not block the location field.

## Order of work

1. When any read returns `truncated: true`, finish the document before reasoning about it.
2. Scan what you have read for the markers above before you record evidence from it.
3. For each marker, work the three places in order and stop at the first resolution.
4. Before submitting, re-check every quote you actually cited: is any of them an interim
   line whose thread you left open? That is the check the 2026-07-26 run failed.
