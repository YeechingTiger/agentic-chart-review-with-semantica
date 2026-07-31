# Boundary cases

Third-level detail: load only when a case looks like one of these. Each entry is the
answer plus the reason, because the reason is what transfers to the next case.

## Behaviour

| Case | Behaviour | Why |
|---|---|---|
| Intraductal carcinoma with focal areas of invasion | `3` | Any invasion, however limited, forces 3. "Focal" does not reduce it. |
| Urothelial carcinoma in situ, no stromal invasion identified | `2` | Confined to epithelium, invasion explicitly looked for and not found. |
| Atypical meningioma invading skull bone | `1` | Meninges invade bone without being malignant. Do not upgrade on spread alone. |
| Metastatic deposit sampled, primary never biopsied | `3` | Metastasis implies invasion. Code the metastasis's histology, the primary's topography. |

## Topography

**Overlapping subsites.** One tumour spanning two or more subsites of an organ with no
identifiable point of origin → subcategory `8`. Several *separate* tumours in different
subsites of one organ → subcategory `9`. These are different situations and the digits are
not interchangeable.

**Specimen site ≠ origin.** A mediastinal node biopsy proving adenocarcinoma of lung origin
codes the **lung** subsite, not `C77`. The node tells you the morphology and that behaviour
is 3.

**When only laterality is documented.** "Right lung" with no lobe is `C349` *only after* you
have searched for a lobe and not found one. Check imaging and operative notes, not just
pathology — see the third mistake in SKILL.md.

## Morphology

**Hedged diagnoses.** `"favor"`, `"consistent with"`, `"suggestive of"` are the pathologist
committing with reservation, not abstaining. Combined with supporting IHC or a matching
clinical picture they support the specific code. Combined with `"pending"`, chase the
addendum first.

**NOS ladder.** `8000` cancer NOS < `8010` carcinoma NOS < `8046` non-small cell carcinoma
NOS < a specific type (`8070` squamous, `8140` adeno, `8041` small cell). Move as far down
this ladder as the record supports, and no further. Coding one rung too high is the common
error; coding one rung too low is fabrication.

**Two reports disagreeing.** Prefer the definitive resection over the initial biopsy, and
cite both. If a consult note restates a finding the primary report contradicts, prefer the
primary report and record the contradiction with `stance=contradicts` — a contradiction that
is not recorded looks identical to one that was never noticed.

## Scope

Out of scope for this skill, answer `SPEC_INSUFFICIENT`:

- Lymphoma, leukaemia, myeloma and other haematopoietic neoplasms — Hematopoietic and
  Lymphoid Neoplasm rules apply instead.
- Cases diagnosed before 2007-01-01, where Solid Tumor Rules do not apply.

## A note on missingness

Missing behaviour is **not** missing at random. It clusters on outside-hospital biopsies and
on declined biopsies, and those groups differ prognostically from the rest. An abstention is
therefore itself a finding worth stating clearly, not a blank to be minimised.
