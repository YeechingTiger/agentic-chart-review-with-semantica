---
name: audit-phi-in-trace
description: >-
  Scan a recorded run's trace for identifiers that should not be on disk — institutional
  person ids, emails, phone numbers, MRN and DOB labels, outbound URLs — and report WHERE each
  one sits, because location decides severity. A script rather than advice; it reads files and
  prints JSON.
license: MIT
metadata:
  category: audit
  kind: script
  entry: scan.py
---

# What a trace already written to disk is carrying

A run leaves a JSONL trace and a JSON manifest. Both are files somebody will read, copy into a
report, or attach to a ticket. This asks the one question nothing else in the system can answer:
**is there an identifier in there, and where.**

Nothing in an object graph can answer it. A run binds one subject and the toolbox holds it, so
another subject is structurally unreachable — but that guarantees the SHAPE of what was read, not
the CONTENT of what was written. An identifier reaches a trace because a document contained it and
a tool result quoted it, and no type prevents that.

## Location decides severity, and that is the whole judgement

The same string means different things depending on which field it landed in:

- **In a model's output, or in an artifact path** — it left the process. IRB-reportable.
- **In a tool argument** — it was sent to a tool this run controls. A warning: worth fixing,
  not worth a report.
- **In a document body a tool returned** — that is the corpus being the corpus. Not a finding.

So a scanner that prints a count is useless. This one prints, per hit, the event, the field and the
location class, and it never prints the value — an HMAC fingerprint instead, so two hits can be
compared without the report becoming the leak.

## What it needs to be told

`ACR_SKILL_TRACE` — the path to a `.jsonl` trace, or to a `.manifest.json` whose sibling trace it
will find. `ACR_PERSON_ID_PATTERN` — the identifier shape at this site; with none set the
institutional-person rule is INERT and the report says so rather than reporting zero.

That distinction matters more than it looks. A scan that finds nothing and a scan that could not
look print the same number, and only one of them is good news.
