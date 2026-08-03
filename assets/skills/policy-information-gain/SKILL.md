---
name: policy-information-gain
description: Use when deciding what to do NEXT rather than what shape the whole traversal should have - before each call, name what is still missing for this field and pick the action that would reduce it most. Tells you what counts as gain and what only feels like it, how to rank the available actions, when the highest-gain action is to stop, and why a locally best path can still fail to support an absence claim. Ranks actions by what they could change about the answer, not by how much of the chart they cover.
slot: policy
license: MIT
---

# Choosing the next action by what it could change

Breadth and depth are shapes: sweep everything, or follow one thread. This is not a shape. It
is a rule applied once per call, and it can produce a wide pass on one chart and a narrow one
on the next, because the ranking is over what is MISSING here rather than over the chart.

## Before each call, three sentences

1. **What is still missing for this field.** Not "what have I not read" — what does the
   contract require that you cannot yet state with a citation.
2. **Which available action would reduce that most.** Name the action and what it would tell
   you if it succeeded AND if it came back empty. An action whose empty result teaches you
   nothing is not a probe, it is a hope.
3. **Whether a class of source that COULD settle this has gone unexamined.** Cheap to ask,
   and it is the one question the other two do not force.

Write them down. A step whose reason you cannot state in these terms is a step taken because
it was available.

## What is gain, and what only feels like it

Gain is a change in what you would answer, or in your grounds for it.

- **High.** A document type the contract lets ESTABLISH the field, when you hold nothing that
  can. A term that separates two readings you are currently unable to decide between. Following
  a deferral to where it was resolved.
- **Near zero, and it is the trap.** A fourth document saying what three already said.
  Re-reading a passage you have cited. Repeating a query you have run. These feel like work and
  move the ledger, so they survive a review that counts documents.
- **NOT near zero, though it looks it: another word for something a term already found.** The
  search folds no synonyms, so a term that hit tells you nothing about what its alternatives
  would reach — including documents that use the other word and never the one you tried. The
  hits you already have are the reason this feels redundant and no reason at all that it is.

Predict what the call will return before making it — being wrong is itself the finding.

But a confident prediction does not by itself make the call worthless. A result you expected
still earns its cost when it gives you a citable source for something you currently only
believe, discharges a proof obligation the contract states, or closes a conflict you have left
open. Gain is the expected improvement in the ANSWER, in the sufficiency of the evidence under
it, in the state of any conflict, or in what remains uncertain — not in your surprise.

## When the highest-gain action is to stop

When no available action would change the answer. That is a claim, and it is checkable: name
what a further search would have to FIND in order to change it, and say where you looked for
that thing. If you cannot write that sentence, you are not finished. If you can, more searching
is spending without a hypothesis.

Running out of ideas for terms is not that sentence. Neither is having read a lot.

## Departing from this, and saying so

You will depart from it — a chart will make the highest-gain action impossible, or the three
sentences will be unanswerable because the contract itself is unclear. That is allowed and it is
not a failure. What is a failure is departing silently, because then a run that worked this way
and a run that ignored it are the same document afterwards.

So when you skip the three sentences, say which one you could not answer and why. "I could not
name what was missing because the contract does not distinguish X from Y" is a finding about the
SPEC, and it is worth more than the step you skipped.

Do not re-plan from scratch on a departure. Backing out and starting over converts one unusable
step into an unusable run, and the step you took is still on the record for whoever reads it.

## What this cannot do

**It optimises one step at a time, so it can walk a path that is locally best and never
establishes that something is ABSENT.** Each individual call was the best available; the set of
them was never chosen to be a denominator. An absence claim rests on what you did NOT read, and
this rule never reasons about that set — `coverage-judgement` does, and it applies on top of
this card whenever your answer says something is not there.

It also decides ORDER, never admissibility. What may establish a field is the contract's, and a
document that ranked highest here is not thereby a witness.
