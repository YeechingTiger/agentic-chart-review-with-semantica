# Marker catalogue: base rates, false positives, resolution

Third-level detail: load when you have found a marker and want to know whether it is worth
steps, or when you cannot find the resolving text. Each entry is the base rate plus the
reason, because the reason is what transfers to the next chart.

Counts below are measured over a 50-patient sample of the real corpus — **7,965 documents**,
scanned 2026-07-26. "docs" is the number of documents containing the phrase at least once.

## How common each marker actually is

| marker | occurrences | docs | where it concentrates |
|---|---|---|---|
| `pending` | 752 | 599 | `Visit-Note` 93, `Primary-Care-MD-OP-Progress-Note` 70, `Discharge-Summary` 65 |
| `addendum` | 551 | 295 | `Emergency-Dept-MD-Progress-Note` 81, `Surgical-Pathology-Document` 28, `Cytology-Report` 19 |
| `final diagnosis` | 283 | 202 | `Surgical-Pathology-Document` 71, `Surgical-Pathology-Report` 37, `Cytology-Report` 36, `FN-Aspirate-Report` 21 |
| `deferred` | 134 | 104 | progress and follow-up notes |
| `see synoptic` | 125 | 169 | `Hem-Onc-MD-OP-Progress-Note` 86, `Surgical-Pathology-Document` 45, `Surgical-Pathology-Report` 14 |
| `preliminary` | 88 | 66 | `Blood-Culture` 15, `Discharge-Summary` 6, `EKG` 6 |
| `immunostain` | 84 | 50 | `Surgical-Pathology-Document` 16, `Cytology-Report` 14 |
| `amended` | 33 | 15 | spread thinly across report types |
| `in consultation` | 31 | 26 | `Discharge-Summary` 8 |
| `outside hospital` | 23 | 14 | ED notes, H&Ps |
| `clinical correlation` | 19 | 19 | pathology and cross-sectional imaging |
| `results pending` | 14 | 12 | ED notes, `FN-Aspirate-Report`, `Cytology-Report` |
| `additional sections` | 14 | 14 | `Cytology-Report` 6 |
| `correlate clinically` | 7 | 5 | radiology only |
| `special stains pending` | **6** | **6** | `Surgical-Pathology-Document` 4 |
| `outside institution` / `outside facility` | 6 / 5 | 4 / 4 | clinic and consult notes |

**Read the top and the bottom of that table together.** `pending` is the second most common
marker in the corpus and is overwhelmingly *not* a lab thread — it is medication refills,
prior authorisations and appointments in clinic notes. `special stains pending`, the phrase
that caused the `8046` error, occurs in **6 documents out of 7,965**.

Two operational consequences:

- **Never search for the long phrase.** On the five real charts reviewed on 2026-07-26,
  `special stains` returned zero hits on four of them, while the stem `stain` returned 8 to
  67. Search `stain`, `addend`, `pend` and read within the document types that can carry a
  lab thread.
- **Filter `pending` by document type before spending steps.** Inside a
  `Surgical-Pathology-Document` or `Cytology-Report` it is a thread. Inside a `Visit-Note`
  it is almost certainly not.

## Density inside the documents that matter

Restricting to the pathology-bearing documents of the five charts reviewed on 2026-07-26
(18 documents), marker counts were:

| patient | markers found in its pathology documents |
|---|---|
| `P05` | `special stain` 10, `addendum` 12, `pending` 4, `preliminary` 2, `immunostain` 2, `correlat` 4 |
| `P02` | `addendum` 14, `correlat` 4, `immunostain` 2, `special stain` 2 |
| `P01` | `addendum` 5, `immunostain` 1, `special stain` 1 |
| `P03` | `pending` 2, `correlat` 2 |
| `P04` | none |

**Four of five charts had an unsettled marker sitting in their pathology.** Assume there is
one until you have looked; do not assume there is one after you have looked and found none.

## Where the resolution was, case by case

| situation | where it turned out to be |
|---|---|
| stains pending, report is long | later in the **same file** — 4 of 18 pathology documents on these charts carry a marker past offset 4,000, the default read window |
| addendum referenced, report has headings | a section named `ADDENDUM 1` / `ADDENDUM 2` — numbered, so an exact lookup for `ADDENDUM` misses |
| addendum referenced, report has no headings | a separate document; the corpus has `Path-Rpt-Addendum`, `Addendum`, `Discharge-Summary-Staff-Addendum` as distinct types |
| `See synoptic report` in a pathology narrative | a **synoptic/CAP-protocol block in the same file**, usually well past the narrative diagnosis. This is the marker that cost a real answer: on 2026-07-27 a run cited a pathology quote ending "Nontumoral lung: Emphysema / See synoptic report", coded histology 8140 from the narrative line "Invasive adenocarcinoma, poorly differentiated", never opened the synoptic block, and passed the gate. The registry coded 8230 (solid adenocarcinoma) — the subtype lives in the synoptic, not the narrative |
| `correlate clinically` in a radiology report | not in radiology at all — the author is disclaiming their own document. Go to the tissue class |
| outside-facility biopsy | nowhere. The document is not in this chart |

## The one marker you cannot resolve

`outside facility` / `outside hospital` / `outside institution` names a document that does
not exist in the record. There is no page to turn to. The correct behaviour is:

1. Search anyway, once — the report is occasionally scanned in under a generic type.
2. If absent, record the reference itself as evidence with `stance="contradicts"`.
3. Abstain on the fields that document would have established, and say plainly that the
   basis is an outside report not present in the chart.

Do not soften this into an inference. A clinic note repeating a diagnosis it attributes to
an outside biopsy is a restatement, not a finding, and treating it as one manufactures
evidence that no one in the care team ever wrote down.

## A note on missingness

Unresolved threads are **not** missing at random. They cluster on outside-hospital work-ups,
on patients who transferred care, and on cases where the sample was insufficient. Those
groups differ from the rest of the cohort, so an abstention driven by an open thread is a
finding worth stating clearly — including *which* marker blocked it, so that the pattern is
countable later.
