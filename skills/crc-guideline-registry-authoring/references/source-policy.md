# Source policy

## Authority order

Use sources in this order for the claim each is designed to support:

1. current official professional guideline or standards-body publication;
2. official guideline summary or implementation tool from the same publisher;
3. current registry standard (CoC STORE, NAACCR data dictionary/SSDI, SEER*RSA);
4. official regulator label or safety communication for drug eligibility/safety;
5. peer-reviewed evidence review for context only.

An NCI PDQ treatment summary is useful evidence context but explicitly is not a formal practice guideline. Do not label it as one.

## Source register contract

Every source entry needs:

- `source_id`
- `authority`
- `title`
- `source_type`
- `version`
- `publication_or_update_date`
- `status`
- `url`
- `accessed_on`
- `scope`
- `use`
- `limitations`
- `license`
- `document_hash`
- `review_status`

Allowed statuses:

- `version_bound`
- `source_pending`
- `superseded_or_update_pending`
- `context_only`

The version can be a named release, DOI publication, manual year, or page-reported update date. “Current” is not a version.

`document_hash` may be `NOT_CAPTURED` in intake, but that value is a promotion blocker rather
than an omission. `review_status` names whether the source binding was agent-checked,
human-checked, or clinically approved; it does not certify the recommendation itself.

## Licensed and living guidance

For licensed guidance:

- register the authority and exact needed scope;
- do not reproduce recommendation content until the authorized artifact is inspected;
- do not use search snippets or another publisher's summary as if it were the licensed source.

For living guidance:

- record both access date and page-reported last update;
- treat a URL without a captured release/update date as `source_pending`;
- repeat source binding before promotion or a new validation run.

## Recommendation extraction

Capture a source anchor that another reviewer can find: recommendation number, table, page, section, or exact named subsection. Store a concise paraphrase and the source's own recommendation type, evidence quality, and strength when supplied.

Do not merge separate recommendations merely because they concern the same biomarker. Testing, interpretation, treatment eligibility, specimen handling, and turnaround time are different rules with different denominators and variables.

## Context completeness gate

A recommendation is blocked when any action-changing context is unknown:

- colon versus rectal;
- localized versus metastatic;
- resectable versus unresectable;
- first line versus previously treated;
- RAS/BRAF/MMR/MSI state;
- tumor sidedness;
- treatment contraindication;
- response state such as clinical complete response.

Use `unknown` only when the source intentionally covers all values. Otherwise use a blocker.

## Copyright-safe notes

Paraphrase recommendation text. Store a short locator, not a copied chapter. The source register may link to an official artifact without vendoring it. Respect institutional access and redistribution terms.
