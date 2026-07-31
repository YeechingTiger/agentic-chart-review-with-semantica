# Staging boundary cases

Third-level detail: load only when a case looks like one of these. Each entry is the answer
plus the reason, because the reason is what transfers to the next case. AJCC 8th edition,
lung.

## Which prefix, and therefore which field

| prefix | when it applies | field in this spec |
|---|---|---|
| `c` | any evidence acquired before the first treatment | `clinical_*` |
| `p` | resection of the primary; `pN` needs nodes removed | `pathologic_*` |
| `yc` / `yp` | staging after neoadjuvant therapy | none — `SPEC_INSUFFICIENT` |
| `r` | restaging at recurrence | none — that is `STORE.1860_1880` |
| `a` | first identified at autopsy | none — `SPEC_INSUFFICIENT` |

**The prefix is a claim about timing, not about who wrote the note.** A surgeon's
pre-operative assessment is `c`; a medical oncologist quoting the resection synoptic
afterwards is `p`.

## T

| Case | T | Why |
|---|---|---|
| Separate tumour nodule, **same** lobe as the primary | `T3` | Same-lobe satellite is a T descriptor, not M. |
| Separate nodule, **different ipsilateral** lobe | `T4` | Still one side, still T. |
| Separate nodule, **contralateral** lung | `M1a` | Crossing to the other lung is metastasis. |
| Visceral pleural invasion, tumour 2 cm | `T2a` | Pleural invasion raises a sub-4 cm tumour to T2a on its own. |
| Main bronchus involved, carina spared | `T2` | Distance from the carina stopped mattering in the 8th. |
| Carina involved | `T4` | With mediastinum, heart, great vessels, trachea, oesophagus, vertebra. |
| Atelectasis of the entire lung | `T2` | Extent of collapse no longer separates T2 from T3. |

**Size ties break upward.** The cut points are 1, 2, 3, 4, 5 and 7 cm; a tumour measured at
exactly 3.0 cm is `T1c`, and one at 3.1 cm is `T2a`. When imaging and the specimen give
different measurements, the specimen governs `pT` and imaging governs `cT` — they are
allowed to disagree, and both are recorded.

## N

| Nodes involved | N |
|---|---|
| Ipsilateral peribronchial, hilar, intrapulmonary — including by direct extension | `N1` |
| Ipsilateral mediastinal or subcarinal | `N2` |
| Contralateral mediastinal or hilar; any scalene or supraclavicular | `N3` |

**`cN0` is an assertion, not a default.** A staging CT or PET that reports node stations is
an assessment and supports `cN0`. A chart with no nodal imaging at all supports `cNX`, and a
chart you did not search supports neither.

## M

| Case | M | Stage group |
|---|---|---|
| Pleural or pericardial nodules, or malignant effusion | `M1a` | `IVA` |
| Contralateral lung nodule | `M1a` | `IVA` |
| Single extrathoracic metastasis, single organ | `M1b` | `IVA` |
| Multiple extrathoracic metastases, one or several organs | `M1c` | `IVB` |

**An effusion is not automatically M1a.** Where the fluid is cytologically negative on
repeated sampling, non-bloody and not an exudate, and the clinician judges it unrelated,
AJCC excludes it from staging. Cite the judgement, not just the effusion.

## Stage groups worth memorising

At `N0 M0`: `T1a` → `IA1`, `T1b` → `IA2`, `T1c` → `IA3`, `T2a` → `IB`, `T2b` → `IIA`,
`T3` → `IIB`, `T4` → `IIIA`. Adding `N1` moves `T1`–`T2` to `IIB` and `T3`–`T4` to `IIIA`.
Any `N2` is at least `IIIA`; any `N3` is at least `IIIB`. Any `M1` is `IV`.

`Tis N0 M0` is stage `0`. `TX N0 M0` — positive cytology with a tumour never visualised — is
`OCCULT`, and it is a clinical group only; there is no pathologic `OCCULT`.

## Summary Stage 2018

Derived from documented extent, never by looking up the AJCC group: `0` in situ, `1`
confined to the lung, `2` regional by direct extension, `3` regional lymph nodes only, `4`
both direct extension and nodes, `7` distant, `9` unknown.

`7` and stage `IV` usually coincide but are established differently — `7` needs documented
distant disease, not a stage group someone wrote down.

## Scope

Out of scope, answer `SPEC_INSUFFICIENT`:

- Post-neoadjuvant `yc`/`yp` staging.
- Small cell lung carcinoma recorded only as limited or extensive stage.
- Lymphoma and other haematopoietic neoplasms (Ann Arbor / Lugano).
- Diagnosis before 2018-01-01, staged under AJCC 7th, where the group definitions differ.

## A note on these examples

Every case above is constructed from the staging manual, not lifted from a chart in this
project — this criterion has not been run against real records yet. That makes them a
statement of the rules, not evidence about where a model actually fails. The four failure
modes in `SKILL.md` earn their place by structural analogy to measured errors on site and
histology; these tables do not, and should be replaced case by case as real ones appear.
