---
name: eval-key-challenge
description: Use when a run's answer disagrees with the recorded answer key and you must establish whether the KEY is derivable from this chart, before anyone concludes the run was wrong. Tells you the three ways a disagreement arises, what evidence separates them, the cheap structural check to run first, and why naming a winner is not your job. Does not decide whether the run was correct - that comparison is the deterministic scorer's and is already done.
license: MIT
metadata:
  slot: eval
  judges: key_derivability chart_support_for_key disagreement_kind
---

# When the answer key and the run disagree, the key is also a suspect

The scorer has told you they differ. It cannot tell you which one is wrong, and neither of the
obvious readings is safe: "the run erred" is the default assumption and it is wrong often enough
to matter; "the key erred" is the flattering one and it is wrong more often than that.

**A registry value is what a person wrote down.** They read outside records you do not have, they
mistype, and they apply a rule differently than the contract does. None of that makes them
careless — an abstractor with the outside hospital's report knows something the chart cannot
show. But it means the key is evidence, not truth, and this repo already refuses to call it gold
until a human has recorded field-level chart derivability and adjudicated.

Your question is therefore NOT "who is right". It is: **can this chart support the key at all?**
That is a question about the chart, it is answerable from the chart, and it is a different
question from the one `==` already answered.

## Start with the structural check, because it is free

Before reading a word of clinical text: **does any document in this chart carry, or fall on, the
value the key names?**

For a date, that is literal — is there a document dated that day, or one whose text contains it.
For a coded value, it is whether any document states the thing the code stands for.

If nothing does, you have a strong finding for the cost of a listing. A key naming a date on
which the chart holds no document at all is not a hard case; it is the cheapest disagreement
there is to detect, and an evaluation that misses it will miss every subtler one.

If something does, that is not yet agreement — the document may not be admissible, or may not
say what the key read into it. Go on.

## The three kinds, and what separates them

Every disagreement is one of these. They are not degrees of the same thing; they have different
owners and opposite remedies, and reporting one as another causes a specific harm.

**KEY_NOT_DERIVABLE_FROM_CHART.** The key is probably right and this chart cannot show it. The
tell is a chart that REFERS to evidence it does not contain: an outside biopsy named in a
transfer note, records requested and never received, a result the patient reports. The run's
abstention was correct reading.
*Report it as this, or:* an abstention gets recorded as a miss, and optimising against that
number teaches the agent to produce values it cannot support — on exactly the subpopulation
where records are incomplete, which is not a random subpopulation.

**KEY_CONTRADICTED_BY_CHART.** The key is wrong. The tell is affirmative: the chart states
something else plainly, and nothing in it supports the key. A date with no document behind it, a
value no document states, a rule the contract spells out and the key did not follow.
*Report it as this, or:* a correct run is filed as a failure. Repeated, the measured accuracy of
a working system decays toward the abstractor's error rate, and the numbers look like a model
problem.

**HUMAN_ADJUDICATION_REQUIRED.** The chart genuinely does not settle it. The tell is that you can
write the case for both readings out of the same documents and neither needs anything the chart
lacks — usually because the contract's rules do not reach the situation.
*Report it as this, or:* naming a winner manufactures certainty from a coin flip, and a spec gap
reported as an agent error never reaches whoever owns the spec.

## What you owe with each verdict

Quote the documents. A verdict of KEY_CONTRADICTED_BY_CHART with no quotation is an opinion about
an abstractor, and it will be read as one. For NOT_DERIVABLE, name the reference to the missing
evidence — the sentence that shows the chart knows something is elsewhere. For ADJUDICATION,
state BOTH readings and the contract sentence that fails to choose between them; if you can only
build one, you have KEY_CONTRADICTED and should say so.

Say `UNDETERMINED` when the chart does not let you decide. It is a real outcome and it is far
cheaper than a confident wrong kind.

## What is not yours

Whether the run was correct. The scorer settled that with `==` before you were called, and
re-deciding it is the one move this whole surface exists to prevent.

Whether the key gets changed. You produce a review obligation for a human. A registry value
becomes something stronger only after a person records the chart answer, the derivability and
the adjudication — that boundary is not yours to cross and nothing you write should read as
though it were already crossed.
