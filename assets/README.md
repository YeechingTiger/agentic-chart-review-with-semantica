# assets/ — everything the framework LOADS

Not code, not output, not data. These are the versioned, content-hashed files the runtime reads
to decide what an answer must mean, how to look for it, and what may judge it. They used to sit
as nine separate directories at the repository root, indistinguishable at a glance from `src/`,
`tests/` and `runs/`.

The code is organised into ten planes under `src/acr/` (see `tests/test_layering.py`). Each
directory here is owned by one of them:

| directory | owning plane | what it is |
|---|---|---|
| `specs/` | contract | task contracts — the fields, the value domains, the proof obligations, the abstention boundaries |
| `codes/` | contract | value-domain code tables, declared as `axes:` — a code system, not necessarily a cancer one |
| `guidelines/` | contract | clinical guidelines the rule engine scores concordance against |
| `contracts/` | contract | JSON Schemas for the spec extensions |
| `skills/` | review + diagnosis | method cards. The `task` / `search` / `general` slots are the chart review agent's; the `eval` slot belongs to diagnosis and never enters a chart run |
| `module_catalog/` | review + audit + evaluation | `runtime_controls/` and `runtime_policies/` are review's, `audit_rules/` is audit's, `evaluators/` is evaluation's — the subdirectories already carry the split |
| `evaluators/` | evaluation | LLM evaluator definitions, loaded by `judge` |
| `pipeline_catalog/` | evaluation | node conditions, dependencies, capability allowlists, budget ceilings |
| `certification_catalog/` | evaluation | must-pass / must-fail fixtures and calibration cohorts |
| `pricing/` | core | `prices.json`, the per-model cost table the spend ceiling is denominated in |
| `usecase/` | usecase | one use case's authoring workspace — the CRC guideline/registry normalisation tranche. **Not part of the framework**; a second use case would sit beside it, not replace it |

Two of these were renamed on the way in, because the old names described something else:

- `audit/` → `pricing/`. It held one file, `prices.json`. Real audit output goes through
  `LocalArtifactStore` and by design never enters the repository, so a directory named after the
  governance plane that contained a price list was the first thing a reader would misread.
- `authoring/` → `usecase/`. 723 YAML files of colorectal-specific normalisation work. The name
  said what someone was doing; the directory needed to say whose it is.

There is deliberately no second level (`assets/contract/specs/`). It would double every path in
the tree and in every command line to say what this table already says.

## Editing a hashed asset

`codes/`, `certification_catalog/` and `usecase/crc/` carry content hashes, and things downstream
refuse to load when the hash and the content disagree. That is not friction to route around: it
is what stops a corrected code table from masquerading as the one a previous run was scored
against. The directory move that created this file tripped exactly that check — a path mentioned
in prose inside a hashed CRC contract — and the right answer was to leave the asset alone, not to
rehash it.
