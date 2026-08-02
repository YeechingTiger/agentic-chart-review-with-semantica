---
name: tactic-query-formulation
description: Use when deciding what to search for, before the first query and whenever searching is not converging. Covers building the term list from the fields the answer must fill rather than from the subject matter, choosing how wide a query should be and reacting to what it returns, and the standing rule that a search which leads to no read has established nothing. Says what to look for; `tool-contract` says how the instrument behaves.
slot: tactic
precondition: "Always applicable: a query has to be formed before anything can be searched."
license: MIT
---

# Deciding what to search for

Terms come from what the ANSWER has to say, not from what the record is about. Everything
below follows from that.

## One column of terms per field the answer must fill

For each field the contract asks you to fill, write the words someone would use when STATING
that field's value. Not words about the subject in general — a record about one thing is full
of words about that thing, and almost none of them are the words that would settle any
particular question about it.

The failure this prevents is specific and easy to walk into: a contract asks for three
different properties, and the term list contains only words for the topic the three share. Then
two of the fields have no column at all, and nothing in the run will notice, because searches
were issued and hits came back the whole time.

**A field with no column is a field you have decided not to look for.** Write the columns down
before the first search.

Two riders. A field whose value is a date or a code needs terms for the EVENT that fixes it,
never for the number itself — the number is what you are looking for, so it cannot be what you
look with. And a field with a short conventional abbreviation needs the long form beside it:
short strings match inside other words and will bury you in hits that are not about your field.

## Choosing how wide a query is, and reacting to what comes back

A query has a width. Too narrow and the local wording defeats it; too wide and it matches
everything and discriminates nothing. Neither end is safe, so choose deliberately and then let
the result tell you which way to move.

- **Many hits, spread across kinds of document that cannot answer** — too wide. Add a word,
  restore an ending, or bound by kind or by date.
- **No hits** — possibly too narrow. Shorten toward the stem, drop a modifier, try the other
  spelling or the abbreviation.
- **Truncated** — you are looking at a slice, not a total, and the shape of what you cannot see
  is not random. `tool-contract` says which end is missing.

Do not reach straight for the shortest possible stem. A stem short enough to be safe against
any wording is usually short enough to match half the record, and the hits it buries are as
lost as the ones a narrow term never found.

## A zero-hit search is a fact about the string you typed

It says that sequence of characters was not found. It does not say the concept is absent: the
record may use another word, another notation, another abbreviation — or may not contain the
document that would have said it at all.

So a zero-hit result is never, by itself, grounds for answering that something is not there.
Widening until you run out of ideas does not change this; running out of ideas is a fact about
you. What an absence claim actually owes is a different question from what a search returns,
and `coverage-judgement` is where it is answered.

## Search locates; reading answers

A hit is a position in a document. Whether the text at that position means what you need is
decided by reading around it — the sentence may be negated, hypothetical, planned rather than
done, or about a different subject than the one you are asking about.

- After at most two searches on one field, open something.
- Record evidence from the document you read, never from the snippet.
- A search that led to no read spent a step and established nothing. Several in a row is the
  signature of refining a query instead of answering a question.

## Re-issuing a query

Issuing the same string twice tells you what it told you the first time. When you find yourself
about to, the thing that needs to change is the term, the width, or the field you are working
on — not the number of attempts.
