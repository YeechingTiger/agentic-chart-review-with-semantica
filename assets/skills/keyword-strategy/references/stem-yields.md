# Stem yields and the per-field worksheet

Third-level detail: load when choosing between candidate terms, when a stem feels too broad,
or when a search is returning nothing. Each entry is the number plus the reason, because the
reason is what transfers to the next chart.

Hit counts measured 2026-07-26 on the five real charts reviewed that day, using the same
substring search the tools expose, with the cap lifted.

## Full term vs stem, per chart

Columns are the last six digits of the patient id.

| term | 071480 | 187360 | 896840 | 327320 | 340680 |
|---|---|---|---|---|---|
| `pathology` | 5 | 58 | 20 | 7 | 41 |
| **`patholog`** | **11** | **91** | **45** | **15** | **75** |
| `bronchus` | 46 | 14 | 13 | 19 | 42 |
| **`bronch`** | **97** | **109** | **15** | **31** | **176** |
| `metastasis` | 7 | 4 | 19 | 0 | 7 |
| **`metasta`** | **30** | **17** | **101** | **19** | **38** |
| `immunohistochemistry` | 0 | 3 | 0 | 0 | 4 |
| **`immuno`** | **6** | **38** | **33** | **0** | **60** |
| `diagnosis` | 32 | 79 | 24 | 22 | 52 |
| **`diagnos`** | **57** | **104** | **33** | **64** | **104** |
| `special stains` | 0 | 0 | 0 | 0 | 7 |
| **`stain`** | **38** | **54** | **21** | **8** | **67** |
| `right upper lobe` | 54 | 4 | 24 | 2 | 54 |
| `upper lobe` | 81 | 90 | 32 | 102 | 60 |
| **`lobe`** | **113** | **153** | **126** | **113** | **71** |
| `addendum` | 6 | 21 | 0 | 0 | 13 |
| **`addend`** | **7** | **24** | **2** | **0** | **17** |
| `biopsy` | 13 | 23 | 21 | 30 | 53 |
| **`biops`** | **13** | **30** | **25** | **30** | **54** |

Where stemming buys nothing — `carcinoma`, `adenocarcinoma`, `specimen` are identical to
their stems on all five charts — leave the word intact. The point is not to truncate
everything; it is that a term with a common inflection or a compound form (`bronchial`,
`metastatic`, `immunostain`, `diagnostic`, `addenda`) loses most of its recall when written
out in full.

Two terms worth noticing for the opposite reason:

- `resection` returned **0** on three of the five charts and at most 5 on the other two. It
  is a registry word, not a chart word; surgeons write `lobectomy`, `wedge`, `excision`.
- `right upper lobe` on `P02` returns 4 while `lobe` returns 153. Word order
  and modifiers in clinical prose are unstable; the head noun is not.

## What the cap does to a broad stem

`search_notes` iterates documents oldest first and returns as soon as `max_hits` is reached
(default 25 through the tool, 40 in the underlying chart API). Measured effect:

| chart | query | hits at cap 25 / uncapped | documents surfaced / matching | what the cap lost |
|---|---|---|---|---|
| `P02` | `patholog` | 25 / 91 | 3 of 18 | both `Surgical-Pathology-Document`s and two `Cytology-Report`s, all on one day |
| `P05` | `stain` | 25 / 67 | 2 of 5 | both same-day `Cytology-Report`s |
| `P03` | `lobe` | 25 / 126 | 5 of 20 | eight months of the span; no pathology lost, by luck |

In the first two rows the capped search reports plenty of hits and looks successful, while
the documents that answer the question are exactly the ones it did not reach. The date span
in the result is the tell: compare the latest date in your hits against the chart's span
from `document_type_summary`. If it stops early, you are looking at a prefix.

Remedies, in order of preference: narrow by `doc_type_contains` to the deciding class, then
by `date_from` around the index event, then raise `max_hits`.

## The per-field worksheet

Fill this in before the first search. One row per field in the spec's `fields` list.

| field | what a clinician writes for its value | stems | short forms to pair |
|---|---|---|---|
| an anatomic site | organ, subsite, laterality, adjacency | `lobe`, `bronch`, `hilar`, `lateral` | `RUL`, `LLL` — three characters or fewer are dropped by the runtime keyword check, so never search only these |
| a tissue diagnosis | the diagnosis line, the microscopic description, stain results | `carcinom`, `squamous`, `adenocarcinom`, `immuno`, `stain` | `IHC`, `NSCLC` |
| a behaviour or invasion fact | how invasion is described or excluded | `invas`, `in situ`, `margin`, `stromal` | — |
| a date of an event | the words for the **event**, not the number | `resect`, `lobectom`, `excis`, `admit`, `cycle` | — |
| a treatment given | agent names, route, administration language | drug stems, `infus`, `administ`, `fraction` | — |
| a status or summary judgement | the summarising document's own vocabulary | `stage`, `recommend`, `discuss` | — |

The test for a finished worksheet: **every field has at least one stem that is not shared
with any other field.** A term list where all the columns are the same is a topic list, and
a topic list is what falsified the keyword obligation on this project.

## Stopping

Stop searching a field when you can name the document that answers it, or when every stem in
its column has been searched, every non-truncated result has been read, and every truncated
result has been narrowed and re-run. Stopping earlier and recording "not documented" makes a
claim about the whole chart on the strength of a prefix of it.
