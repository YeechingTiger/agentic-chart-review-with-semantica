# Field design: what becomes a field, and what the field will accept

Third-level detail. Load when you are deciding where one spec ends and the next begins, when
you are writing a `format` or an `allowable_values` list, or when a value domain has to admit
an unknown code.

Every example is from a spec in `specs/`. Two of them are defects that are still there.

---

## 1. Splitting versus merging

### The two failures, which are not symmetrical

**Merging what has different evidence hides the error you most need to see.**

Clinical stage and pathologic stage are different registry data items. Different evidence
(everything known before first treatment, versus the resection specimen), different timing
(fixed at the start of treatment, versus whenever the specimen is cut), different owner (the
treating clinician, versus the pathologist). `STORE.700_880` therefore declares nine fields,
not one `stage`.

The reason is not tidiness. The commonest error in this criterion is copying one into the
other — reading `cM0` out of a pathologic synoptic, or filling `clinical_stage_group` from the
resection. With a single `stage` field that error is not a wrong value; it is a value the
output shape cannot distinguish from a right one. No check can fire on a distinction the
schema does not carry, so the error becomes invisible rather than rare.

**Splitting what one pass answers produces three answers to one question.**

Site, histology and behaviour are one spec, `STORE.400_522_523`, because one agent pass over
the same pathology report answers all three. Three specs would be three passes over the same
chart, and nothing constrains them to agree. `registry_catalog`'s module docstring is about
exactly this: a user asks for `primary_site`, the system runs `STORE.400_522_523`, and
resolution is per spec rather than per variable so that the three answers are one answer.

### The three questions, in the order that decides it

| | merge | split |
|---|---|---|
| would one pass over the same documents answer them all? | yes | no |
| do they share evidence rules, timing and owner? | yes | no |
| **what wrong answer do you most fear?** | anything but a value crossing between them | a value crossing between them |

Row three breaks the ties, and it is the row that gets skipped. `pathologic_m` exists as its
own field with `format: "pM1(a|b|c)?"` so that `pM0` — which is `cM0` having walked across the
boundary — is a rejection rather than a plausible answer.

### Cases already decided in this tree

| spec | shape | why |
|---|---|---|
| `STORE.400_522_523` | 3 fields, 1 spec | one pathology report answers site, histology and behaviour together |
| `STORE.700_880` | 9 fields, 1 spec | nine data items, one staging conversation; the c/p split is the point |
| `STORE.390` | 2 fields | `month_day_imputed` records *how* the date was obtained, not another date |
| `STORE.1860_1880` | 2 fields | a recurrence date is meaningless without the type that says a recurrence happened |
| `STORE.610` | 1 field, never filled | not a property of the chart at all — see `skills/store-to-spec/references/proof-obligations.md` |

`month_day_imputed` is worth copying as a pattern. When a rule says "approximate it if you
must", the approximation is a second fact about the answer, and a boolean beside the value is
where it goes. Burying it in prose loses it at the first aggregation.

### One constraint from outside the spec

A field name that reaches two specs is an **error** in `registry_catalog`, naming both — never
first-wins, because nothing declares an ordering over specs. So `primary_site` may be produced
by exactly one spec in the tree. If a second criterion needs it, that criterion consumes the
first spec's output; it does not re-declare the field.

---

## 2. Value domains that enforce a rule for free

`check_field_formats` runs on every submitted answer, for every criterion, with no per-spec
configuration: `format` via `re.fullmatch`, `allowable_values` via exact string membership
after `strip()`. Both are already in the spec, so anything you can push into them is enforced
at zero authoring cost — and enforced on the run that would otherwise have been accepted and
stamped `gate_validated`, which is what happened to `primary_site="C3412"`, four digits
against a declared `C\d{3}`, before the checker existed.

### Which of the two to use

| use | when | example |
|---|---|---|
| `allowable_values` | a closed enumeration you can write out | behaviour `["0","1","2","3"]`; summary stage `["0","1","2","3","4","7","9"]` |
| `format` | a structured string with internal grammar | `cT(X\|0\|is\|1mi\|1a\|1b\|1c\|1\|2a\|2b\|2\|3\|4)` |
| both | never — they are AND-ed, and one of them will be the stale one | |

### Rules the domain can carry

| rule | how it is encoded | what it stops |
|---|---|---|
| there is no `pM0` | `format: "pM1(a\|b\|c)?"` | `cM0` copied across the c/p boundary |
| there is no `cMX` | `cM(0\|1a\|1b\|1c\|1)` | "unknown" hidden behind an X instead of taken as an abstention |
| occult carcinoma is clinical only | `OCCULT` in `clinical_stage_group`, absent from `pathologic_stage_group` | a pathologic group that AJCC does not define |
| a c-category is not a p-value | the `c`/`p` prefix is inside each pattern | `pT3` submitted for `clinical_t` |
| topography is three digits | `C\d{3}` | `C3412` |

The common thread: each is a property of the **code system**, decidable from the string alone.
A rule that depends on the case — "a pathologic T requires that the primary was resected" — is
not encodable here. It belongs in `decision_rule`, and in an `answer_checks` entry if it is
mechanically decidable from the cited evidence.

### Four traps, all of which have bitten

- **Registry notation is not a pattern.** `format: "CCYYMMDD"` accepts exactly one string:
  `CCYYMMDD`. Still live in `STORE.390` and `STORE.1860_1880`.
- **YAML eats the backslash.** `format: "C\d{3}"` in double quotes is a `ScannerError` and the
  whole spec fails to load. Write `"C\\d{3}"`, or single-quote it, or leave it plain
  (`format: C\d{3}`, as the ablation spec does). The loud failure is the good case here; the
  bad case is a pattern that survives with the escape stripped.
- **`fullmatch` is not casefold.** `ct2a` is rejected against `cT(...)`. That is deliberate —
  the `c` prefix is the datum — but decide it rather than inherit it, and say so in
  `description` so the model emits the right case.
- **Quote every enumeration entry.** An unquoted `99` beside a quoted `"IIIA"` splits the
  domain into two types; the comparison is `str(v)`, so it happens to work until something
  else reads the YAML and gets an int.

Alternation order is *not* a trap under `fullmatch`: the engine backtracks, so
`cT(1|1a)` accepts `cT1a`. It matters the moment the same pattern is used with `re.match` or
`re.search`, which return `cT1` from the same input. The shipped specs order longest-first
anyway; keep doing it, and understand it as insurance rather than as a fix.

### Leaving a field null must stay free

`check_field_formats` skips `None` and `""` — "absent is abstention's business, not the format
checker's". That is what makes a strict domain workable: a nine-field staging spec against a
chart that documents three of them must be able to leave six null at no cost. If null were
expensive, every strict pattern would become an argument for guessing.

---

## 3. NOS ladders and their proof burden

### Three different unknowns, routinely conflated

| shape | claim | discharged by |
|---|---|---|
| NOS (`C349`, `8000`, `8010`, `8046`, `cT2` where `cT2a` exists) | "the more specific value is not documented" | having searched for the specific values, in every stratum that may mention them |
| X (`cTX`, `cNX`, `pTX`) | "this was assessed and could not be determined" | a document showing the assessment happened and was indeterminate |
| unknown code (`99`, summary stage `9`, recurrence `99`) | "documented nowhere in this record" | the full coverage proof — this is `EVIDENCE_INSUFFICIENT` wearing a code |

None of the three is a default, and the third is the one that gets used as one. `cNX` is not
the code for "I did not read the staging CT"; `99` is not the code for "no stage in the
pathology report".

### The ladders

Morphology, lung, most specific last:

```
8000 cancer, NOS
  8010 carcinoma, NOS
    8046 non-small cell carcinoma, NOS
      8070 squamous cell carcinoma · 8140 adenocarcinoma · 8041 small cell carcinoma
```

Each rung is a positive claim that no rung below it is documented. `8000` and `8010` are
**not** interchangeable: if the physician wrote "carcinoma", `8010` is the honest rung and
`8000` is a claim that even "carcinoma" is absent. A hedge is not a rung — "favor squamous
cell carcinoma" is a commitment with a reservation and codes `8070`, which is why the shipped
`not_less_specific` check lists `favor squamous` in `contradicted_by`.

Topography, lung:

```
C349 lung, NOS
  C341 upper lobe · C342 middle lobe · C343 lower lobe · C340 main bronchus
```

with two subcategory conventions that are easy to invert: `8` is one tumour overlapping two or
more subsites of unknown origin, `9` is NOS — and `9` is also what you must write when
multiple tumours occupy different subsites of one organ.

Staging has the same structure without the NOS vocabulary: `cT2` where AJCC 8th only defines
`cT2a` and `cT2b`; a stage group `99` where a group is stated in a tumour board note.

### Writing the checks

Two of the three `answer_checks` kinds exist for this. Both are cheap; write them for every
unknown-shaped value the domain admits.

```yaml
answer_checks:
  - field: primary_site
    kind: nos_requires_search
    nos_values: ["C349"]
    required_searches: ["lobe", "bronchus"]
    message: "C349 asserts no subsite is documented. Search for it before claiming it is absent."
  - field: primary_site
    kind: not_less_specific
    nos_values: ["C349"]
    contradicted_by: ["upper lobe", "middle lobe", "lower lobe", "RUL", "RML", "RLL"]
    message: "A documented lobe gives a specific subsite; C349 asserts the subsite is unknown."
```

`contradicted_by` is a **flat list per entry**, so write one entry per NOS value rather than
one per field. Folding `cT1` and `cT2` into a single entry makes a quote containing `ct1a`
reject a correctly coded `cT2`.

Where the specific value lives is usually not where the NOS value came from. `C349` came from
pathology that said only "Right Lung"; `right upper lobe` was in imaging and oncology notes.
So the search list must point at the documents that carry the *specific* value, not at the
ones that carry the field.

### Missingness is not a nuisance parameter

Unknown codes cluster: on outside-hospital biopsies, on declined biopsies, on patients who
never reached an oncology consultation, on fragmented care. Those groups differ prognostically
from the coded population, so a downstream analysis that drops them is not dropping noise.
Record it in `special_codes_not_mar`, and put the consequence in `downstream_warning` —
`STORE.610` does this for Class of Case `00`, `38` and `49`, all of which produce zero or
undefined survival time and none of which is caught by a range check.
