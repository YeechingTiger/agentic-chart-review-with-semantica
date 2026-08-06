---
name: attribution-method
description: Use when explaining why a completed chart-review run produced the outcome it did. Gives the order to investigate in - target event first, then trace, then exact spec rules, then a discriminating probe only if the trace cannot separate rival causes - and the rules for when a cause may be called primary. Does not decide whether the run was clinically right; that needs truth the packet may not carry.
license: MIT
metadata:
  category: eval
---

# attribution-method

You are explaining ONE recorded run. You do not re-run extraction and you never edit a
specification.

## The order, and why it is this order

1. **List the target events and select exactly one.** This is the outcome you are explaining.
   Everything after this is about that one event; a finding that would not have changed it is not
   an explanation of it.
2. **Read the trace before you open the chart.** `inspect_trace` first, always. What the run
   searched, read, cited, submitted, and why it stopped is the subject. Opening the chart first
   makes you an extractor forming your own view, and then you are comparing the run against
   yourself rather than explaining it.
3. **Open the exact rule with `inspect_spec`.** Do not paraphrase a rule you have not opened. A
   cause stated against a remembered rule is a cause stated against nothing.
4. **Probe only when the trace cannot separate rival causes.** `open_attribution_probe` needs at
   least two named alternatives and a concrete discriminator — the thing you expect to differ
   between them. A probe with one alternative is a confirmation, not a discriminator.
5. **Say how each cause relates to the target.** A genuine defect that would not have changed the
   selected event is `UNRELATED_DEFECT` and cannot be primary, however real it is.
6. **Record a bounded counterfactual test.** `LIKELY` or `CONFIRMED` needs a SUPPORTED test for the
   selected target. Without one, downgrade to `POSSIBLE` or `UNRESOLVED`.
7. **Take the skeptic review seriously.** If it returns REVISE or UNRESOLVED, the report is
   UNRESOLVED. It is not a formality to be argued past.
8. **Run one final challenge probe** with `confirmation=true`. If it exposes a conflict, downgrade.
9. **Finish only with `submit_attribution`.**

## Two failure modes worth naming

**Explaining the run you would have done.** The question is why THIS run produced THIS outcome, not
what a careful reviewer would have found. A cause the run could not have acted on is context, not
cause.

**Promoting a plausible story.** Rival causes that the evidence cannot separate stay rivals. An
`UNRESOLVED` report that names two live hypotheses and the discriminator that would settle them is
worth more than a confident one that picked the more narratable of the two.
