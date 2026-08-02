# What we saw on the development corpus, when it was a policy card

Until 2026-08-02 all of this sat inside `assets/skills/keyword-strategy/SKILL.md` and was
sent to the model as instruction. It is not instruction. It is a set of observations about
one synthetic corpus, one spec, and thirty-nine runs of an earlier tool surface, and inside
a policy it reads as a rule that only holds where the same thing happens to be true.

Kept because it is the raw material a CERTIFIED experience asset would be built from —
`cli label scan` then `cli assets evolve/certify/adopt`. Nothing has been through that
pipeline yet, so nothing here has a held-out gain or a permutation control behind it.

Two of its claims were checked against the code on 2026-08-02 and are WRONG: the snippet is
250 characters, not 160, and no three-character minimum exists anywhere in `src/`.

---

## The card as it stood

```markdown
---
name: keyword-strategy
description: Use before issuing the first search and whenever searching is not converging - deciding what terms to search for a spec with more than one output field, a search returning zero hits, a search coming back truncated true, issuing several similar queries without opening a document, or being about to record a not-documented, unknown or NOS value. Covers building the term list from the output fields rather than from the topic, stemming instead of synonym sprays, what the hit cap silently hides, and the rule that a search which leads to no read did nothing.
slot: search
license: MIT
---

# Searching a chart so that it finds things

Three rules, each of which was broken on a real chart in this project on 2026-07-26 by a
model that had the search hints in its prompt. Variable-agnostic.

## 1. Build the term list from the FIELDS, not from the topic

One column of terms per output field. For each field, write the words a clinician would use
to state that field's **value** — not words about the subject matter in general.

Measured. The site/histology/behaviour spec declares three fields, one of them an anatomic
location. Its eight search hints are `pathology, biopsy, final diagnosis, specimen,
carcinoma, adenocarcinoma, resection, cytology` and its stratum keyword list is five of the
same. **Not one names a lobe, a laterality or a bronchus** — yet the spec's own answer check
refuses the NOS site code unless `lobe` and `bronchus` were searched. The term list was
built from the topic ("this is a cancer-coding task") instead of from the fields.

The consequence, across the 39 runs in `runs/`: only 17 searched any lobe term and 6 any
bronchus term, while the two most-issued queries in the whole set were `final diagnosis`
(28 times) and `adenocarcinoma` (27). Patient `P03` was coded lung-NOS having
run no site search at all, on a chart where `lobe` matches 126 times.

Practical form, before the first search:

```
field: primary_site   -> lobe, bronch, lateral, right, left, upper, lower, hilar, apical
field: histology      -> patholog, carcinom, adenocarcinom, squamous, small cell, cytolog
field: behavior       -> invas, in situ, margin, stromal
```

A field with no column is a field you have decided not to look for. Two riders: short
abbreviations (`RUL`, `LUL`, `FNA`) are dropped by the runtime's keyword machinery at three
characters or fewer, so always pair them with a longer term; and a field whose value is a
date or a code needs terms for the **event**, not for the number.

## 2. Stem, do not spray synonyms

Search is a case-insensitive substring match. A stem therefore subsumes every inflection at
no cost, and a longer phrase can only ever match less. Measured hit counts on the five real
charts (full term vs stem, same chart):

| full term | hits | stem | hits |
|---|---|---|---|
| `immunohistochemistry` | 0 – 4 | `immuno` | 6 – 60 |
| `bronchus` | 13 – 46 | `bronch` | 15 – 176 |
| `metastasis` | 0 – 19 | `metasta` | 17 – 101 |
| `pathology` | 5 – 58 | `patholog` | 11 – 91 |
| `special stains` | 0 on four of five charts | `stain` | 8 – 67 |
| `right upper lobe` | 2 – 54 | `lobe` | 71 – 153 |

The spray is worse than useless because it consumes the budget that reading needs. In the
worst run measured, `tumor`, `tumour`, `malignant`, `malign`, `cancer`, `benign`, `in situ`,
`dysplasia`, `polyp`, `adenoma`, `squamous`, `sarcoma`, `lymphoma`, `grade`, `stage` and
`TNM` each returned **zero** hits; `biopsy` returned 1 and `biops` returned 15; and the
highest-yield query of the run — a two-word site term, 18 hits — was issued 50th out of 50.

`carcinoma` was issued as query 4 and again, character-for-character, as query 30 of the
same run. Re-issuing a query is the signature of not refining one.

## 3. Search locates; reading answers

Measured over all 39 traces: **337 searches, of which 225 (67%) were never followed by a
read of any document they hit, and 134 (40%) returned zero hits.** One run issued 50
searches (48 distinct) and performed 3 reads. A search that leads to no read did nothing
except spend a step.

- After at most two searches on one field, open a document.
- A hit is a location, not a fact. The snippet is 160 characters of context and is not
  citable evidence on its own — read the document around it before recording it.
- Zero hits is a result about **your vocabulary**, not about the chart. Shorten to the stem,
  drop a modifier, try the abbreviation and the expansion, or check the type filter. Only
  after the stem also returns nothing is absence evidence of absence.

## The hit cap hides the newest documents

`search_notes` scans documents **oldest first and stops the moment it reaches `max_hits`**
(default 25). A broad stem on a busy chart therefore returns the oldest chatter and silently
omits everything after it — including, routinely, the report you were looking for.

Measured: on patient `P02`, `patholog` at the default cap surfaces 3 documents
of the 18 that match and **loses all four pathology reports** — both
`Surgical-Pathology-Document`s and two `Cytology-Report`s, all dated the same day — because
the cap is exhausted inside the first three weeks of a year-long span. On
`P05`, `stain` at the default cap loses both same-day `Cytology-Report`s.

`truncated: true` on a search result means you are looking at the oldest slice of the
matches. Thirty search results in the traces came back truncated. React by narrowing with
`doc_type_contains` or `date_from`, or by raising `max_hits` — never by reasoning over the
slice as though it were the whole.

## Order of work

1. Write one term column per output field before searching anything.
2. Reduce each term to its shortest unambiguous stem.
3. Search the highest-value field first, narrowed by document type where triage identified
   a deciding class.
4. Open a document after every one or two searches. Record evidence from the document, not
   from the snippet.
5. Before recording any not-documented, unknown or NOS value, check that every term in that
   field's column was actually searched and that no result was left truncated.

Measured stem-yield tables and the per-field worksheet:
`assets/skills/keyword-strategy/references/stem-yields.md`.
```

---

## Its stem-yield reference

```markdown
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
```
