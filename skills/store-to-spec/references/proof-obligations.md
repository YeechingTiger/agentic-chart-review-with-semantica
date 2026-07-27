# Proof obligations: witnesses, coverage, and honest abstention

Load this reference when deciding which document classes can establish a field, what must be
searched before an unknown or negative answer is allowed, or whether a proposed coverage gate
can ever pass.

This document describes the current runtime, not an aspirational schema. `ExtractionSpec`
allows unknown keys, so a plausible-looking YAML key can load and hash while changing no
behaviour. Use only the keys and policies named here, and keep the distinction between what
the YAML records and what the runtime actually enforces.

The shipped specs have not been reviewed by a registrar. Treat every `model_authored`
proof policy, threshold, and document grouping as a draft until that review is recorded.

## Contents

1. [Two different proofs](#1-two-different-proofs)
2. [Classify evidence per field](#2-classify-evidence-per-field)
3. [What positive-witness YAML enforces today](#3-what-positive-witness-yaml-enforces-today)
4. [Negative coverage that is enforced](#4-negative-coverage-that-is-enforced)
5. [Implemented stratum policies](#5-implemented-stratum-policies)
6. [Implemented gate switches](#6-implemented-gate-switches)
7. [Make the obligation satisfiable](#7-make-the-obligation-satisfiable)
8. [Unknown and NOS values need negative proof](#8-unknown-and-nos-values-need-negative-proof)
9. [`not_applicable` and the two abstentions](#9-not_applicable-and-the-two-abstentions)
10. [Authoring checklist](#10-authoring-checklist)

## 1. Two different proofs

A positive and a negative answer make different claims.

| answer | proof | claim earned |
|---|---|---|
| `FOUND` | witness | at least one cited document supports the submitted value |
| `EVIDENCE_INSUFFICIENT` | coverage | the defined evidence universe was worked and did not establish a value |
| `SPEC_INSUFFICIENT` | applicability failure | this spec or its configured data source cannot answer this case |

A witness does not prove coverage. A gate-validated negative does. The emitted answer reflects
that difference:

- an accepted `FOUND` is labelled `proof_basis: WITNESS` and carries no
  `coverage_attested`;
- an accepted `EVIDENCE_INSUFFICIENT` is labelled
  `negative_basis: GATE_VALIDATED` and carries the computed coverage ledger;
- a budget exhaustion or give-up carries no coverage attestation and routes to a human.

Do not require a positive answer to finish a negative-coverage gate. That turns a cheap
witness into an unnecessary chart census. Do not accept a negative merely because no witness
was found. That turns retrieval failure into evidence of absence.

## 2. Classify evidence per field

For every `(document class, field)` pair, assign one of three meanings:

| class | meaning | normal treatment |
|---|---|---|
| `can_establish` | one document of this class can settle this field by itself | review exhaustively when a negative is contemplated |
| `may_mention` | it may restate, qualify, or point to the answer but is not the preferred independent authority | search, read every hit, then sample misses |
| `cannot_establish` | this class is declared incapable of establishing this field | validate that exclusion by forced sampling |

These are field-level relations, not permanent properties of a document type. Imaging may
establish tumour location while only mentioning histology. A treatment note may establish
that a drug was administered while only mentioning a molecular result copied from another
report.

Write `establishes:` explicitly on every stratum when a spec has more than one field:

```yaml
proof_obligation:
  for_negative:
    mode: stratified_exclusion
    strata:
      - name: can_establish
        establishes: [primary_site, histology]
        match: {doc_type_matches: ["Pathology"]}
        policy: exhaustive
      - name: may_mention
        establishes: [primary_site]
        match: {doc_type_matches: ["Imaging", "Oncology"]}
        policy: search_then_read_hits_and_sample_misses
        required_keywords: ["primary site"]
        min_sample_of_misses: 25
      - name: cannot_establish
        establishes: []
        match: {rest: true}
        policy: validate_by_sampling
        min_sample: 25
```

This is a fragment, not a loadable spec: every enforced element in a real spec also needs its
`provenance` record.

Assignment is first-match-wins. The runtime evaluates non-`rest` strata in declaration order
and moves `rest: true` strata to the end. `doc_type_matches` is a case-insensitive substring
test, not an exact type match or a regex. Test routing against the actual document-type
vocabulary.

An empty `establishes` list means “no fields” to a human reviewer. Do not rely on the older
code comment that calls empty “all fields”; the current negative gate does not use
`establishes` to decide its verdict at all.

## 3. What positive-witness YAML enforces today

The binding-shaped grammar is:

```yaml
proof_obligation:
  for_positive:
    statement: "One authoritative report supports each non-null field."
    witness:
      primary_site: [can_establish, may_mention]
      histology: [can_establish]
```

`ProofObligation.witness_strata` parses this mapping, and the review surface displays it.
`establishes` is also parsed, content-hashed as an enforced element, and copied into coverage
results.

However, the current `gate_answer` implementation does **not** compare a FOUND citation's
document stratum with `for_positive.witness`, and it does not check that the cited stratum's
`establishes` list contains the answered field. Today, FOUND acceptance enforces only:

1. at least one recorded evidence item exists;
2. every non-null value passes `format` or `allowable_values`;
3. configured `answer_checks` pass.

Therefore:

- author `witness` and `establishes` so the intended contract is reviewable and ready for the
  runtime connection;
- do not report per-field witness admissibility as mechanically enforced;
- add a test that exposes the missing enforcement if the distinction matters to the variable;
- do not use a stratum name as a substitute for the missing check.

`exhaustive_until_witness` also does not prove a semantic witness. Its current completion rule
is merely `reviewed > 0`; it does not inspect whether the reviewed document was cited or
established the field. Use it only with that limitation stated.

## 4. Negative coverage that is enforced

For `EVIDENCE_INSUFFICIENT`, `gate_answer` performs work in this order:

1. Compute keyword matches among already drawn validation documents without an LLM.
2. Resolve sampled documents the agent has read. A cited sampled document counts as relevant;
   an uncited one counts as non-relevant. Keyword-matched but uncited documents are reported
   as suspected recognition failures but do not fail the gate.
3. Draw any outstanding forced samples. The agent cannot choose those documents.
4. Evaluate the configured gate against the coverage ledger.
5. Require that the patient document list was opened.
6. Require every top-level `for_negative.required_keywords` term to have been searched.

Forced sampling validates the **document-class exclusion or keyword retrieval strategy**. It
does not validate reading comprehension: a relevant sampled document that the model reads but
fails to cite is recorded like an irrelevant one. Treat a zero-hit sample as evidence about
the retrieval design, never as proof that the model understood every sampled note.

Only `proof_obligation.for_negative.required_keywords` is checked directly by
`graph.check_gate`. Stratum-level `required_keywords` participate through the
`search_then_read_hits_and_sample_misses` result. `required_doc_types_read` is parsed and
rendered in the prompt but is not enforced by the current gate.

Keyword discharge is one-directional: the required term must occur inside a term actually
searched. Searching `invasive carcinoma` covers required `carcinoma`; searching `carcinoma`
does not cover required `final diagnosis`. Avoid required terms that are accidental
substrings of broader search terms, and make every required term reachable from
`search_hints`.

## 5. Implemented stratum policies

Use only these four policy names:

| policy | current completion semantics |
|---|---|
| `exhaustive` | complete only when every document assigned to the stratum was read |
| `exhaustive_until_witness` | complete when at least one assigned document was read; semantic witness is not checked |
| `validate_by_sampling` | runtime draws up to `min_sample`; relevance is inferred from citation |
| `search_then_read_hits_and_sample_misses` | all required searches must run, every hit must be read, and the runtime samples up to `min_sample_of_misses` misses |

Do not author `exhaustive_per_window`. It appears in a shipped spec but has no implemented
completion branch and receives no pending sample draw, so it cannot be worked as its name
implies. Window functions exist in `coverage.py`, but the normal `check_gate` call does not
pass windows to `evaluate_gate`; no window gate is currently active.

`max_tolerated_hits` is parsed into `StratumSpec` but no current gate reads it. The effective
tolerance is hard-coded:

- `cannot_establish` passes exclusion validation only with at least one sampled document and
  zero sample hits;
- a `may_mention` miss sample validates only with zero miss-sample hits.

Do not claim a nonzero tolerance by writing `max_tolerated_hits`.

## 6. Implemented gate switches

These keys under `proof_obligation.for_negative.gate` are read:

| key | enforced meaning |
|---|---|
| `require_can_establish_nonempty` | a stratum named `can_establish` is declared; patient-level `N > 0` is **not** required |
| `per_claim_can_establish_nonempty` | currently the same global name-presence check, not a true per-claim check |
| `exhaustive_strata_complete` | the result named `can_establish`, if present, is complete |
| `exclusion_validated` | the result named `cannot_establish`, if present, has at least one sample and zero hits |
| `required_keywords_all_searched` | no stratum-level required keyword remains unsearched |
| `keyword_list_validated` | the result named `may_mention`, if present, satisfies its search/read/sample rule |
| `max_elusion_upper` | the largest non-`can_establish` Clopper-Pearson upper bound is at or below the cap |

When a switch is omitted, `evaluate_gate` defaults
`exhaustive_strata_complete`, `exclusion_validated`,
`required_keywords_all_searched`, and `keyword_list_validated` to true.

Do not author `all_claims_satisfied` as though it were enforced. `claims[].strata` are
flattened, and results are indexed by stratum name. Reusing `can_establish` across claims can
overwrite an earlier result during gate evaluation. True per-claim satisfaction is not
implemented. Prefer one non-claim stratification until this runtime path is repaired.

The `confidence` value inside `for_negative` is not passed into `CoverageLedger` by the normal
agent path; the ledger currently uses its default `0.95`. Do not imply that changing this YAML
value changes the bound.

## 7. Make the obligation satisfiable

Test both an empty ledger that must fail and a fully worked ledger that must pass. Also test
small strata, because census-sized draws are not treated as zero elusion in every policy.

At 95% confidence, a zero-hit sample of 25 has one-sided Clopper-Pearson upper bound:

```text
1 - 0.05^(1/25) = 0.1129
```

Thus `min_sample: 25` can satisfy `max_elusion_upper: 0.12`, while a lower cap cannot.
Increasing a YAML cap does not repair an invalid document classification; a hit in
`cannot_establish` means that document class must be promoted and the design rerun.

Check these edge cases:

- A declared but patient-empty `can_establish` stratum is complete and satisfies
  `require_can_establish_nonempty`; that switch tests the spec design, not whether this
  patient has an authoritative document.
- A patient-empty `cannot_establish` stratum cannot satisfy `exclusion_validated`, because
  the gate requires at least one sample.
- A small nonempty sampling stratum may be read in full yet still exceed
  `max_elusion_upper`, because the current calculation uses the binomial bound rather than
  recognising a census.
- A `may_mention` stratum with zero misses can satisfy its miss-sample count, but only after
  all declared required searches ran and every search hit was read.
- An empty `required_keywords` list makes a `may_mention` keyword-list validation fail.
- A searched hit that is never opened is covered by neither the hit review nor the miss
  sample and therefore fails validation.
- Duplicate stratum names collapse in gate evaluation.

For a concrete dry run, use a synthetic chart such as P01 and assert the work sequence, not
only the final verdict: list documents, run every required search, read all exhaustive
documents and search hits, read the runtime-drawn samples, then resubmit.

## 8. Unknown and NOS values need negative proof

An unknown-shaped code is a positive value carrying a negative claim:

- NOS says no more-specific value was documented;
- X says the concept was assessed but could not be determined;
- `99` or another registry unknown says the value was documented nowhere in the scoped
  record.

The runtime currently sends any such submitted code down the FOUND path. The negative
coverage gate is therefore **not automatically invoked**. Add the implemented
`answer_checks` that apply:

- `nos_requires_search` rejects a configured NOS value until its configured searches ran;
- `not_less_specific` rejects a configured NOS value when cited evidence contains a listed
  more-specific phrase.

These checks are narrower than a full coverage proof. They do not exhaust strata or force
samples. If the registry meaning truly asserts chart-wide absence, decide whether the
unknown code should be represented as an abstention instead of a FOUND value, or document
the remaining proof gap explicitly.

P02 illustrates the difference: a report saying “assessed, indeterminate” may witness an X
code; merely failing to find a result does not.

## 9. `not_applicable` and the two abstentions

Choose `mode: not_applicable` when no clinical-note document class could establish the field
in principle. This is a property of the specification and data source, not a finding that
this patient happens to lack the right document.

Current runtime limitation: `mode` itself is not branched on by `gate_answer` or
`check_gate`. For a variable that truly lives outside notes, `data_source: outside_notes` is
the implemented control: finalisation forces `SPEC_INSUFFICIENT` and assigns
`remedy_class: WRONG_DATA_SOURCE`.

Therefore:

- use `mode: not_applicable` to state the proof design honestly;
- use `data_source: outside_notes` when the configured source is in fact wrong;
- do not expect `mode: not_applicable` alone to force an abstention;
- never create fake strata and searches for a variable no note can establish.

Keep the abstentions distinct:

- `SPEC_INSUFFICIENT`: the variable is outside the configured source, the case is excluded,
  or the specification cannot decide it;
- `EVIDENCE_INSUFFICIENT`: the specification applies, the chart was covered as required,
  and the evidence did not establish a value.

P03 having no pathology report is normally `EVIDENCE_INSUFFICIENT` for a pathology-dependent
field. A field that exists only in a facility registration system is
`SPEC_INSUFFICIENT` when the agent has notes alone. The first is a patient-record finding;
the second is a source-contract failure.

## 10. Authoring checklist

Before shipping a proof obligation:

1. List every field and classify each real document class separately for that field.
2. Route actual document-type strings through `StratumSpec.matches`; remember substring and
   first-match semantics.
3. Declare `establishes` on every stratum, but label it as review metadata until FOUND
   witness enforcement is connected.
4. Use only the four implemented policies and the seven implemented gate keys above.
5. Make every required keyword reachable from `search_hints`.
6. Compute whether the sample size can meet `max_elusion_upper`.
7. Exercise empty, small, and ordinary strata.
8. Assert that an empty ledger fails and a fully worked ledger passes.
9. Test NOS and unknown codes as negative claims, not convenient defaults.
10. Use `not_applicable` only for a specification/data-source impossibility; use
    `EVIDENCE_INSUFFICIENT` for a covered chart that lacks evidence.
11. Record provenance for every enforced format, value domain, answer check, match,
    `establishes`, required keyword, sample threshold, and gate threshold.
12. State plainly which intended controls are review-only rather than runtime-enforced.
