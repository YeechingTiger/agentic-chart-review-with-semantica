---
name: eval-missed-evidence
description: Use when the answer key says a value is documented but the run reported it absent or wrong, and you must find why the text was never reached. Tells you the four places a retrieval failure can sit, how to distinguish never-searched from searched-and-missed from read-and-misjudged, and what evidence from the trace each conclusion requires. Does not establish that the value is in the chart - that comes from the answer key and the scorer.
slot: eval
judges: [retrieval_failure_locus, term_coverage, type_filter_effect]
license: MIT
---

# The answer was in the chart and the run did not use it

**Confirm from the scorer and the answer key that the value is genuinely documented before you
start.** A run that reported absence on a chart that truly lacks the value is correct
behaviour, and diagnosing it as a miss is how correct abstention gets trained away.

## Four places the failure can sit

Work them in order; each is ruled out by different evidence in the trace.

1. **Never searched.** No term the run issued could have matched the text. Evidence: the list
   of terms in the trace, and the text that carries the value. If no term is a substring of
   the surrounding line, stop here — the rest is moot.
2. **Searched, filtered out.** A term that would have hit was issued with a document-type
   filter that excluded the document holding it. Evidence: the search call's filter argument
   and the document's type.
3. **Hit, not read.** The search returned the document and the run did not open it. Evidence:
   the search result list and the absence of a matching read call. Look at what it opened
   instead and in what order — this is usually a ranking problem.
4. **Read, not used.** The document was read in full and the value was not extracted, or was
   extracted and then discarded. Evidence: the read call, and the reasoning that follows it.
   This is not a retrieval failure and must not be reported as one; it belongs to whoever owns
   the task contract's evidence standing.

## What each conclusion owes

Every locus you name requires the specific trace evidence listed beside it. A conclusion of
"never searched" with no term list quoted is a guess. If the trace does not contain what you
need — for instance a truncated read whose extent is unrecorded — say the locus is
undetermined and say what would settle it.
