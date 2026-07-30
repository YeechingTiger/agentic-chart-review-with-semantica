---
name: store-to-spec
description: How to turn a registry standard item (CoC STORE, NAACCR, AJCC) into an extraction spec in this repo's specs/*.yaml format. Use when adding a variable the system can extract - deciding which data items become one spec and which must stay apart, writing value domains the runtime can actually enforce, scoping a stratum to the fields it speaks to, choosing what must be proved before an absence claim is accepted, and recording which lines came from the manual and which the model invented. Its sibling guideline-to-rules writes RULES over variables; this one writes the VARIABLES those rules are keyed on.
license: MIT
---

# Turning a registry item into an extraction spec

A spec is the agent's contract: the decision boundary and the evidentiary rules, deliberately
not the navigation path. It is loaded by `acr.spec.ExtractionSpec`, rendered into the prompt,
stratified by `acr.coverage`, gated before an answer is accepted, and content-hashed so that
two labels are only comparable when they were produced under the same `spec_hash`.

Everything below is a mistake that was made in this repository, by a competent model with the
rules in its context, or by the engineer who wrote the spec. Two of them are still in the tree
as you read this.

`ExtractionSpec` is declared `extra="allow"`. A key you invent will load, hash into the
spec_hash, render nowhere and enforce nothing — silently. Nothing warns you. That is the
background condition for every mistake on this page.

## 1. `format` is a Python regex, always

`check_field_formats` applies it with `re.fullmatch`. Registry notation put there compiles
cleanly as a pattern and then matches exactly one string — its own text:

```yaml
fields:
  - name: date_of_initial_diagnosis
    format: "CCYYMMDD"        # rejects 20100612 and every other valid date
```

This historical defect shipped in `STORE.390.date_of_initial_diagnosis` and
`STORE.1860_1880.first_recurrence` before being repaired. It raised nothing:
`check_field_formats` swallows `re.error` so a typo cannot block a patient run, and
`CCYYMMDD` is not an error — it is a valid pattern that happens to accept no date. Every
submitted answer was rejected with a message blaming the value.

The notation belongs in `description`, where the model reads it and the runtime does not:

```yaml
fields:
  - name: date_of_initial_diagnosis
    type: string
    format: "(19|20)\\d{2}(0[1-9]|1[0-2]|99)([0-2]\\d|3[01]|99)"
    nullable: false
    description: "CCYYMMDD; 99 in month or day where the registry standard permits it"
```

Note what that pattern had to accept: `20100499`. It is the answer to one of the spec's own
boundary cases — a diagnosis documented only as "the spring of 2010". A value domain that
rejects the spec's own worked example is the same defect as `CCYYMMDD` wearing better clothes.

**Write one value the pattern must accept and one it must reject, for every field, into the
test file.** `test_every_declared_format_is_a_python_regex_not_registry_notation` states the
reason: an unexercised pattern is indistinguishable from `CCYYMMDD`.

## 2. A stratum name is not a claim — scope it with `establishes:`

`STORE.400_522_523` answers three fields from one agent pass and had one stratification
serving all three. Imaging fell in the `rest` stratum, named `cannot_establish`. That name is
true of histology and behaviour and **false of primary_site** — the same spec says outright
"Radiology can localise a mass; it cannot establish histology or behaviour."

Patient P03 was coded `C349` (lung NOS) while "right upper lobe" sat in seven other note
types. The pathology said only "Right Lung". The prose said radiology was admissible for
topography; the architecture told the agent those documents were useless; the agent believed
the architecture.

```yaml
strata:
  - name: cannot_establish
    establishes: [primary_site]     # NOT histology, NOT behavior — and it is not silent
    match: {rest: true}
    policy: validate_by_sampling
```

The rule: a stratum name is prose for the reader, `establishes:` is the claim. When they
disagree the agent follows the name. So for every (stratum, field) pair, ask whether a
document of that class can establish **that field** — not the criterion, the field. The
answers differ per field, which is the entire subject of
`skills/store-to-spec/references/proof-obligations.md`.

## 3. Split or merge: one data item, or one agent pass?

Both directions have failed here, and they fail differently.

**Merging what has different evidence hides the error.** Clinical stage and pathologic stage
are different registry items with different evidence (pre-treatment workup versus the
resection specimen), different timing, and different owners. A single `stage` field makes the
commonest error in the criterion — conflating them — *unrepresentable*, and therefore
invisible to every check that could have caught it. `STORE.700_880` declares nine fields
because staging is nine data items.

**Splitting what one pass answers produces disagreement.** Site, histology and behaviour live
in one spec because one agent pass over the same pathology report answers all three. Three
specs would mean three passes over the same chart that can return three disagreeing answers
for one patient — the failure `registry_catalog`'s module docstring describes and the two
ledgers `state.py` had to lose.

The question that decides it is not "are these conceptually one thing?" but:

| ask | merge | split |
|---|---|---|
| would one pass over the same documents answer them all? | yes | no |
| do they share evidence rules, timing and owner? | yes | no |
| what wrong answer do you most fear? | *not* one field copied from another | one field copied from another |

That third row is the operative one. If the error you fear most is a value crossing from one
field to the other, splitting them is what makes the crossing expressible, and only then can
a check see it: `pathologic_m` with `format: "pM1(a|b|c)?"` rejects `pM0` precisely because
`pM0` is `cM0` that has walked across the boundary.

## 4. Encode the rule in the value domain, where enforcement is free

There is no `pM0` and no `cMX` in AJCC 8th. So the domains simply refuse them:

```yaml
fields:
  - name: clinical_m
    format: "cM(0|1a|1b|1c|1)"
    description: "AJCC 8th clinical M, e.g. cM1b. There is no cMX; leave null if undetermined."
  - name: pathologic_m
    format: "pM1(a|b|c)?"
    description: "Only pM1, pM1a, pM1b, pM1c exist. There is no pM0 and no pMX; leave null."
```

This costs nothing to run, needs no per-spec configuration, and applies on every criterion.
More importantly it closes the exit: an agent that wants to say "unknown" must leave the field
null and take the abstention, rather than hide behind an X that reads downstream as a value.
`OCCULT` is in `clinical_stage_group`'s `allowable_values` and absent from
`pathologic_stage_group`'s for the same reason — occult carcinoma is a clinical group only.

Two limits.

- Encode properties of the **code system**, never of the case. "A pathologic T requires a
  resection" is not a property of the string `pT2a`; it belongs in `decision_rule` and in an
  `answer_checks` entry. A regex that tries to express it will reject a legal value.
- Quote every entry in `allowable_values`. An unquoted `99` beside a quoted `"IIIA"` is how a
  domain silently becomes two types; `check_field_formats` compares `str(v)`, so the bug
  surfaces somewhere else entirely.

## 5. NOS, X and 99 are positive claims, not defaults

`C349`, `8000`, `cTX`, `cNX`, stage group `99`, summary stage `9`, recurrence type `99`: each
one asserts that the specific value **is not documented in this chart**. That is an absence
claim, and it carries the same proof burden as `EVIDENCE_INSUFFICIENT` — with none of the
visibility, because it arrives dressed as an answer.

Measured on 2026-07-26: `C349` was submitted having run zero searches, with the lobe
documented in seven other note types.

Two of the three `answer_checks` kinds exist for this and cost six lines each. Write both for
every unknown-shaped value your domain admits, or record in the spec why you did not:

```yaml
answer_checks:
  - field: clinical_stage_group
    kind: nos_requires_search
    nos_values: ["99"]
    required_searches: ["stage", "tnm"]
    message: "99 is a claim of absence and carries the same burden as EVIDENCE_INSUFFICIENT."
```

The NOS *ladder* — which unspecified code sits above which, and why `8000` and `8010` are not
interchangeable — is in `skills/store-to-spec/references/field-design.md`.

## 6. Cite the item number you verified, or say you did not

The numbers in `spec_id` follow this repository's **file-naming convention**. They have never
been reconciled against the NAACCR data dictionary. `STORE.700_880` admits this in its own
`source_authority.note`, and that note is the only place in the tree where it is admitted.

```yaml
source_authority:
  document: "AJCC Cancer Staging Manual 8th edition (Lung) + CoC STORE 2025 (updated 2025-04-24)"
  items: ["TNM Clin T / N / M and Clinical Stage Group"]      # names, verified in the manual
  note: "The numeric identifiers in spec_id follow this repository's file-naming convention."
```

Nothing in the runtime reads `source_authority`. It exists for the human who has to approve
the spec — which is exactly why a wrong number there is caught by nothing, and why the honest
form is to name the items you checked against the manual and to state plainly that the digits
in the filename are not among them.

## Order of work

Write the sections in this order. Each is unwritable before the one above it, and every
inversion below has produced a specific defect in this tree.

1. **question** — one sentence, in the words a registrar would use, naming the tumour or event
   it is about. If you cannot write it without "and", you are probably holding two criteria.
2. **fields** and their value domains — names, types, `format`, `allowable_values`,
   `nullable`. Do this second because everything after it names a field: strata scope
   themselves to field names, and a check names a value. Decide the split-or-merge question
   here (§3) and encode what the code system forbids (§4).
3. **evidence_rules** — `counts_as_evidence` and `does_not_count`, written as document
   classes rather than as topics. `does_not_count` is the half that is usually missing and is
   where the interesting content is: it is the sentence "imaging cannot establish histology"
   that later becomes a stratum's scope.
4. **conflict_rules** — `if:`/`then:` pairs for the disagreements the record actually
   contains: two reports, a summary line less specific than the body, a restatement
   contradicting a primary source. Each should say which source wins *and* that the loser is
   still cited.
5. **proof_obligation** — the mode, then the strata, then the gate. Choose the mode by asking
   whether any document class could establish the answer *in principle*: if none can, the mode
   is `not_applicable` and no amount of searching will change that (`STORE.610`). Scope every
   stratum with `establishes:` (§2). Check the gate is satisfiable before you ship it: 25
   documents sampled with zero hits gives a Clopper-Pearson upper bound of 0.1129, so a
   `max_elusion_upper` below that is unpassable no matter how much work is done. Only
   `proof_obligation.for_negative.required_keywords` is enforced by the gate, and every term
   in it must be reachable from `search_hints`, or the agent is asked to prove something it
   was never told to look for.
6. **abstention** — both states, distinctly. `SPEC_INSUFFICIENT` is "this specification does
   not cover this case"; `EVIDENCE_INSUFFICIENT` is "the specification is clear and this chart
   is not". Collapsing them turns a spec defect into a patient finding.
7. **answer_checks** — one per mechanically decidable rule you already wrote above. The
   clinical content stays in the spec where a registrar can review it; the checker contains no
   oncology. If a rule needs judgement, leave it in `decision_rule` and do not fake a check.
8. **provenance** — see below. Last, because it is a statement about everything above it.

## Write down what you did not get from the manual

Every element you inferred, generalised or invented is `model_authored`. That is the default:
an element with no entry in the spec's `provenance:` map is listed, by id, in the "what we made
up" section of the document a registrar is asked to sign.

```yaml
provenance:
  rule.1: source_authority        # transcribed from the manual
  rule.4: model_authored          # explicit is better, but silence means this anyway
```

Marking is a key, not a tone. Hedging in prose — "possibly", "generally" — does not mark
anything; it just makes a fabricated rule sound like a cautious real one.

**The reason this rule exists is a fact about this repository, not a principle.** The four
original specs — `STORE.390`, `STORE.400_522_523`, `STORE.610`, `STORE.1860_1880` — were
written by a model in a single commit from manual text in its context, committed under a human
author's name. `STORE.700_880` was written the same way. **No registrar has read a line of any
of them.** Boundary cases, code lists, conflict rules and thresholds in those files are
indistinguishable, from the outside, between transcription and invention.

A spec is not clinically usable until a registrar has read its model_authored list. Say that
where the next person will see it, and keep the list complete: capping it at ten items with
"and 47 more" is the exact softening the section exists to refuse.

## Before you ship it

Give the spec its own test file, and pin these five, all of which have failed before:

- every `format` accepts at least one real value and rejects at least one wrong one;
- `establishes:` names only fields that exist, and the establishing stratum speaks to all of
  them;
- the document types you enumerated route where you think they do — `doc_type_matches` is a
  case-insensitive **substring**, which files `Fine-Needle-Report` outside `["Pathology"]` and
  sweeps `Speech-Language-Pathology-Note` in;
- the gate FAILS on an empty ledger and PASSES once the obligation is worked, in the same test
  file, so that neither result is vacuous;
- no required keyword is a substring of another — the gate matches bidirectionally, so
  `"stage"` would silently discharge `"pathologic stage"` and vice versa.

Further reading:

- `skills/store-to-spec/references/field-design.md` — splitting versus merging, value domains
  that enforce rules for free, NOS ladders and their proof burden.
- `skills/store-to-spec/references/proof-obligations.md` — choosing strata for a new variable:
  which document classes can single-handedly establish it, which may merely mention it, which
  are silent, and why those three sets are different for every field.
