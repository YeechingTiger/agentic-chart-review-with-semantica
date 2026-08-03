"""Give each staged repository a pyproject, a README, a .gitignore and a first commit.

    python tools/scaffold_repos.py --out /tmp/acr-split
    python tools/scaffold_repos.py --out /tmp/acr-split --push --owner <gh-user>

THIRD-PARTY DEPENDENCIES ARE COMPUTED, NOT LISTED. Every `import` in every file of a staged repo is
resolved: standard library is dropped, `acr.*` routes to a sibling distribution, anything else is a
third-party requirement pinned to the range the source repository already used. Hand-listing them is
how a repo ends up declaring `langchain` because its neighbour needed it — and how it ends up NOT
declaring `pyyaml` because it was already installed everywhere the author looked.

THE READMEs ARE WRITTEN, not generated. Every one of these repositories is meant to be usable on a
task that is not cancer-registry abstraction, and a generated README cannot make that case: it would
describe files. What a reader needs is what the thing does, what it deliberately does not do, what
crosses its boundary, and the one constraint it learned the hard way — because that last part is the
reason to use it rather than write it again. The inventory at the bottom is generated; everything
above it is prose.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from split_repos import REPOS

#: Version ranges, taken verbatim from the source repository so nine repos cannot drift into nine
#: opinions about what version of langchain works.
PINS = {
    "deepagents": "deepagents>=0.6.12,<0.7",
    "langchain": "langchain>=1.3.11,<2",
    "langchain_openai": "langchain-openai>=1.4.1,<2",
    "litellm": "litellm>=1.55",
    "langgraph": "langgraph>=1.2.5,<1.3",
    "pydantic": "pydantic>=2.9",
    "yaml": "pyyaml>=6.0",
    "typer": "typer>=0.15",
    "rich": "rich>=13.9",
    "pytest": "pytest>=8.0",
    "langchain_core": "langchain-core>=0.3",
}

#: Imported, but NOT a requirement, and each for a different reason worth naming rather than
#: silently allow-listing:
#:   scipy      — inside `try/except` in `coverage.clopper_pearson_upper`, with a pure-Python
#:                bisection fallback right below it. Optional by construction.
#:   lc_callback — resolved at runtime from a deployment directory (`site.AUDIT_DIR`) inside a
#:                `try/except` that records WHY it failed. Not a package.
#:   openpyxl   — used by a one-off script that ships inside an asset directory, not by the package.
#:   mcp, anyio — only reachable from `mcp_server.main`, which is the `acr-mcp` console script.
#:                Declared as an OPTIONAL extra so `pip install acr-chart-review` does not pull an
#:                MCP stack for a user who only wants the CLI.
#:   conftest, hooks_harness — test-local modules that travel with the tests that import them.
NOT_REQUIRED = {"scipy", "lc_callback", "openpyxl", "conftest", "hooks_harness"}
OPTIONAL_EXTRAS = {"acr-chart-review": {"mcp": ["mcp>=1.0", "anyio>=4.0"]}}

STDLIB = set(sys.stdlib_module_names)

#: Per repo: the paragraph that says what it is for anyone who is not doing cancer registry work,
#: what it refuses to do, and the one thing it knows that a rewrite would not.
PROSE: dict[str, dict[str, str]] = {
    "acr-chart-review": {
        "tagline": "Run one declarative contract over one document corpus, leave a record that "
                   "distinguishes a correct answer from a lucky one, and compare arms over it.",
        "what": """A **task contract** is a YAML file stating a question, the numbered decision rules
that may establish an answer, the conflict rules that order them when two apply, the evidence rules
that say what counts as support, and the closed set of outcomes a run may conclude. The agent
receives one, plus a corpus of documents for one subject and prose method cards assembled by slot,
and gets seven tools: list, summarise-by-type, search, read, read-batch, record-evidence, submit.

It leaves a JSONL trace of every call and a JSON manifest stamped with enough identity that two runs
are comparable — the contract's hash, the corpus's content hash, the code SHA, the resolved arm.

This repository holds the agent, the contract vocabulary it runs on, the synthetic corpus it is
tested against, and the experiment harness that compares arms over it. Those four move together: a
change to what a contract may say changes the agent, the charts that exercise it and the numbers the
harness prints, in one commit.

The loop itself is off the shelf (deepagents / LangGraph). Everything here sits on top of it.""",
        "generalises": """Substitute the corpus and the contract and nothing here knows the
difference. Three assumptions: the documents are text with a type and a date, the question is
answerable from them, and somebody will later need to know WHY the answer was what it was. Benefits
eligibility, contract review, incident triage, regulatory conformance, systematic-review screening —
the word "chart" is in this package's history, not in its interfaces.""",
        "not": """- Does not score itself. That is `acr-eval`, in another distribution on purpose: a
  runtime that can see the answer key is a runtime that will use it.
- Does not edit its own prompts, and does not derive its own priors. That is `acr-improvement`.
- Does not know what an identifier looks like at your site. `acr/core/site.py` asks, and refuses to
  proceed with a real corpus configured and no identifier shape declared.""",
        "hard_won": """**Most checks are advisory, and that was measured rather than chosen.** Five
deterministic content checks were removed after destroying 58 correct values against 21 helps. The
coverage gate went advisory after ~150 rejections of which 27 refused the reference value's exact
tuple. A thread refusal went advisory at 28% reference-destroying. Every time, a rule that was right
about the text was wrong about the answer more often than it was right.

**Of eleven wrong answers in the last valid batch, ZERO were retrieval failures.** `NEVER_LOOKED 0`,
`READ_NOT_CITED 0` — the agent opened the document carrying the answer every single time and got the
reading wrong. Measure that before investing in better search; `tools/measure_controller_value.py`
is the script.

**The corpus is a repository concern, not a `.gitignore` concern.** Six held-out charts were once
generated into a tree whose ignore file carried `*[0-9].txt`, written for iCloud conflict copies
(`Name 2.txt`) but missing the space — and a document filename ends in a digit. Already-tracked
files were unaffected so nothing looked wrong, and the six went into version control with ONE file
each. 1,589 documents were never committed, which inverted the one guarantee a held-out set exists
for.""",
    },
    "acr-eval": {
        "tagline": "Score, judge, attribute and audit COMPLETED runs — from the record alone.",
        "what": """Four things that must not become one thing:

- **Score** against an answer key, deterministically, with detectors for what a score cannot see:
  zero documents read, a search that cannot fail, a rejection loop.
- **Judge** what no rule can score, fenced — a judged number is an OPINION and is refused wherever a
  deterministic evaluator exists.
- **Attribute** a wrong answer to a cause, by an agent that has never been shown the key.
- **Audit** the trajectory truth-blind: did this run touch a subject it was not reviewing, did an
  artifact leave its boundary, does a trace already on disk carry an identifier.

Everything reads finished manifests and traces. Nothing re-runs anything.""",
        "generalises": """Any agent leaving a structured trace can be scored, judged, attributed and
audited by this package. `RunRecord` is the interesting part — it pairs a manifest with its trace
and is schema-tolerant across its own history's drift; every accessor on it is a scar from a field
that moved.""",
        "not": """- Never rewrites an answer. Three independent guards enforce that, because the
  first version of this plane did.
- Never claims a conclusion above its truth mode. GOLD, REGISTRY_REFERENCE and BLIND are a CEILING,
  verified against the recorded runs.
- Does not fix anything. Routing a finding to an owner is `acr-improvement`.""",
        "hard_won": """**This plane's own accuracy has never been measured, and the code says so.**
`meta_evaluate_attributions` requires 30 adjudicated cases and macro-F1 0.80 and has never run;
there are 2 attribution records on disk. Treat its output as a hypothesis until that number exists.
An evaluation plane nobody has evaluated has an unknown error rate, and printing that is more honest
than implying otherwise.

**An inert rule and a clean rule print the same zero.** `audit-phi-in-trace` reports
`inert_rules` and `person_id_pattern_configured` alongside its counts, because "nothing is there"
and "we could not look" are different results. One audit rule in this tree could never fire from the
day it was written — it read four trace keys that nothing writes — and no count would have said
so.""",
    },
    "acr-improvement": {
        "tagline": "Learn from finished runs and change what the next one is given.",
        "what": """Two outputs, and the difference between them is the point:

- **A prior the agent is GIVEN.** Read every document of a development set once, cheaply, against ONE
  requirement. Price each candidate term by what it actually retrieves. Write a retrieval plan.
  Certify it on a held-out set before anything at scale may use it. The result is an INPUT to a run.
- **A change to the system ITSELF.** Every text parameter an agent reads is a parameter: the system
  prompt, the method cards, the clauses of the contract. Take classified failures, route each to the
  parameter that could have caused it, propose an edit, and require paired validation before it
  stands.

`BehaviorSignature` reduces a run to what it answered, what it cited, which rules it claimed and how
it got there, hashed — so "these two runs behaved the same" is a comparison and not an
impression.""",
        "generalises": """Any prompted system you are trying to move deliberately, and any retrieval
task where the vocabulary of the question and the vocabulary of the corpus differ — which is most of
them. The routing is the substance: without it an optimiser reverse-engineers a story from the
outcome and starts confidently rewriting rules that were never at fault.""",
        "not": """- Never applies an edit. It proposes; validation is a separate decision.
- Never edits a domain rule on its own authority. A semantic change needs gold AND human
  adjudication; a REGISTRY_REFERENCE truth mode can only produce a question for an expert.
- Never certifies on the set it developed on.""",
        "hard_won": """**A term derived from an answer is not evidence that the term works.**
`answer_leak.py` guards it because the failure is invisible: a keyword list derived from labelled
data scores beautifully on that data and adds nothing anywhere else.

**And a caution about scope.** Two of the six contracts this was built against have since had their
keyword lists and strata REMOVED on purpose, because measurement showed the agent did better
choosing its own terms. The deriving path refuses at the door rather than inventing assets for a
contract that declares none — which is correct, and also means it may be solving a problem your task
does not have. Measure retrieval reachability first; `tools/measure_agency.py` in
`acr-chart-review`.""",
    },
    "acr-rules": {
        "tagline": "The rules a domain expert owns: authored so they can read them, and executed so "
                   "conformance is decided by rule and never by a model.",
        "what": """Two halves of one job, and they are here together because they share a
dependency — the intake router resolves a question against the same variable catalogue the
conformance engine reads.

- **Authoring.** An arbitrary question routes to a contract that can answer it, or to an explicit
  statement of what is missing — never to a guess. Eleven formal completeness checks in four tiers
  that cost four different things to run and mean four different things when they pass. Then: render
  the contract as a document a domain expert reads in ten minutes and marks up, and record, with a
  content hash, that a named person approved one element AS IT WAS WORDED THAT DAY.
- **Conformance.** A rule engine with no model anywhere in its import closure. Read variables
  somebody else extracted, evaluate a guideline over them, return conformance or not — and for a
  non-conforming case, narrow to which of four causes survives: the care itself, the documentation,
  the extraction, or a justified exception.""",
        "generalises": """Wherever the person who owns the rules cannot read the file the rules live
in — which is most regulated domains — and wherever conformance over extracted structured data has
to be defensible: clinical guidelines, regulatory checklists, contractual obligations, eligibility
rules. The dependency direction is the useful part: it can say which variables a rule reads, and
what breaks if one of them moves.""",
        "not": """- Reaches no model in the conformance engine. Not a default — a property, and the
  import closure is the proof.
- Does not extract anything. It consumes an extract.
- Does not decide whether a rule is RIGHT. It makes the rule legible and records who said it was.
- Does not collapse the four causes into one number.""",
        "hard_won": """**Assent to a sentence is not assent to whatever that sentence is edited
into.** Sign-off records carry the element's content hash for exactly that reason, and the next
render reports the approval as withdrawn the moment the wording changes.

**Four causes, and they must not become one number.** A case can be non-conforming because the care
was wrong, because the care was right and the note is silent, because the extraction missed it, or
because a documented exception applies. Two of those are about the subject and two are about the
pipeline. Report them apart or the metric gets optimised by improving extraction and nothing else.

**Five of the eleven lint checks produce zero findings over all six shipped contracts.** They are
regression guards in a passing state, not dead code — but do not read a clean lint as coverage. The
14 tier-1 failures the shipped contracts DO produce are real and unfixed.""",
    },
}


def third_party(repo_dir: pathlib.Path) -> list[str]:
    """Every non-stdlib, non-`acr` import in the staged tree, pinned to the source repo's range."""
    found: set[str] = set()
    for f in repo_dir.rglob("*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            names = ([n.module] if isinstance(n, ast.ImportFrom) and n.module and not n.level
                     else [a.name for a in n.names] if isinstance(n, ast.Import) else [])
            for m in names:
                top = m.split(".")[0]
                if top and top not in STDLIB and top != "acr":
                    found.add(top)
    extras = {n for names in OPTIONAL_EXTRAS.get(repo_dir.name, {}).values() for n in names}
    extra_tops = {n.split(">")[0].split("=")[0].replace("-", "_") for n in extras}
    out, unknown = [], []
    for t in sorted(found):
        if t in NOT_REQUIRED or t in extra_tops or t == "tools":
            continue
        if t in PINS:
            out.append(PINS[t])
        else:
            unknown.append(t)
    if unknown:
        print(f"    !! UNPINNED third-party imports in {repo_dir.name}: {', '.join(unknown)} — "
              f"add to PINS or to NOT_REQUIRED with a reason")
    return out


def inventory(repo_dir: pathlib.Path) -> str:
    rows = []
    for d in sorted({p.parent for p in repo_dir.rglob("*.py")
                     if "tests" not in p.parts and "__pycache__" not in p.parts}):
        py = [p for p in d.glob("*.py")]
        if not py:
            continue
        rows.append(f"| `{d.relative_to(repo_dir)}/` | {len(py)} |"
                    f" {sum(len(p.read_text(encoding='utf-8').splitlines()) for p in py)} |")
    ntest = len(list((repo_dir / "tests").glob("*.py"))) if (repo_dir / "tests").is_dir() else 0
    body = "\n".join(rows) or "| — | — | — |"
    return (f"| directory | modules | lines |\n|---|---|---|\n{body}\n\n"
            f"{ntest} test file(s).")


def write_repo(name: str, repo_dir: pathlib.Path, sha: str, owner: str | None) -> None:
    spec, prose = REPOS[name], PROSE[name]
    deps = third_party(repo_dir)
    sib = [f'"{d}"' for d in spec["deps"]]
    dep_block = ",\n    ".join([f'"{d}"' for d in deps] + sib)

    is_library = (repo_dir / "src" / "acr").is_dir()
    pkg_block = ("""# `acr` is a PEP 420 implicit namespace package, shared across the nine repositories that came out
# of one tree. There is no `src/acr/__init__.py` in ANY of them, on purpose: adding one here would
# shadow every sibling's subpackage and the failure looks like a missing module.
[tool.setuptools.packages.find]
where = ["src"]
include = ["acr*"]
namespaces = true
""" if is_library else """# NOT A LIBRARY. This repository ships data and scripts, not an importable package, and declaring
# an empty module list is how that is said out loud rather than discovered when a build backend
# fails looking for `src/`. `pip install -e ".[dev]"` still works and installs only the test tooling.
[tool.setuptools]
py-modules = []
""")
    extras = OPTIONAL_EXTRAS.get(name, {})
    extra_block = "".join(
        f'{k} = [{", ".join(chr(34) + v + chr(34) for v in vs)}]\n' for k, vs in extras.items())

    (repo_dir / "pyproject.toml").write_text(f'''[project]
name = "{name}"
version = "0.1.0"
description = "{prose['tagline']}"
readme = "README.md"
requires-python = ">=3.12"
license = {{ text = "MIT" }}
dependencies = [
    {dep_block}
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]
{extra_block}

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

{pkg_block}
[tool.pytest.ini_options]
testpaths = ["tests"]
''', encoding="utf-8")

    (repo_dir / ".gitignore").write_text('''# Virtualenvs. Both forms: a bare `.venv/` pattern does not match a SYMLINK named `.venv`, and
# committing one replaced a real environment with a self-referential link once already.
.venv
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
*.egg-info/
dist/
build/

# Credentials, and anything derived from real data.
.env
*.key

# Run output. Never committed: it is derived from the corpus a deployment points at, and this
# project's rule is that patient-derived artifacts live outside every checkout.
runs/
''', encoding="utf-8")

    install_note = ("""`acr` is a [PEP 420](https://peps.python.org/pep-0420/) implicit namespace package. This
distribution ships `src/acr/<plane>/` and **no `src/acr/__init__.py`** — nine repositories install
into the same `acr.*` namespace, so `from acr.contract.spec import load_spec` resolves whichever
sibling provides it. Adding an `__init__.py` at the namespace root shadows every sibling, and the
failure presents as a missing module rather than as a conflict."""
                    if is_library else
                    """**This repository is data and scripts, not an importable package.** The install
step brings in the test tooling and nothing else; run the scripts from the checkout. The siblings
listed below are what you need installed for them to do anything.""")

    fixtures = """
### Shared fixtures

The document corpus lives in `acr-corpus` and the task contracts and method cards live in
`acr-chart-review`. They are DATA, found by path rather than imported, so they are not dependencies
— clone the siblings beside this repository and everything resolves:

```
parent/
  acr-corpus/          <- corpus/patients/
  acr-chart-review/    <- assets/specs/, assets/skills/
  {name}/              <- you are here
```

Or point at them explicitly, which is what a deployment with its own corpus does:

```bash
export ACR_CORPUS=/path/to/documents
export ACR_SPECS=/path/to/contracts
export ACR_SKILLS=/path/to/cards
```

`acr.core.site.corpus_root()` resolves the environment variable first, then the path under the
current directory and each parent, then a sibling checkout — and raises naming the variable when
nothing is found, rather than letting a missing directory surface later as a puzzling
`UNKNOWN_PATIENT`."""

    url = f"https://github.com/{owner}/{name}" if owner else f"<owner>/{name}"
    siblings = "\n".join(
        f"- [`{d}`]({url.rsplit('/', 1)[0]}/{d}) — {PROSE[d]['tagline']}" for d in spec["deps"]
    ) or "None. This is the bottom of the stack."

    (repo_dir / "README.md").write_text(f'''# {name}

**{prose['tagline']}**

{prose['what']}

## Where else this applies

{prose['generalises']}

## What it does not do

{prose['not']}

## What it learned the hard way

{prose['hard_won']}

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

{install_note}
{fixtures}

## Depends on

{siblings}

## Contents

{inventory(repo_dir)}

## Provenance

Extracted from a single repository at `{sha[:12]}` on 2026-08-03. History was not carried:
`git subtree split` could only reach back to the commit that reorganised seventeen root directories
into seven, so per-plane history was one to twenty-one commits and blame would have pointed at a
move rather than at a decision. The archive repository holds the full history, and this tree's
reasoning lives in its docstrings, which travelled with the files.

License: MIT.
''', encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", action="store_true", help="create the GitHub repo and push")
    ap.add_argument("--owner", default=None)
    ap.add_argument("--public", action="store_true",
                    help="create public repos; default is private")
    args = ap.parse_args()

    out = pathlib.Path(args.out).expanduser().resolve()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()

    for name in REPOS:
        d = out / name
        if not d.is_dir():
            print(f"  !! {name} not staged; run split_repos.py first")
            return 1
        write_repo(name, d, sha, args.owner)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=d, check=True)
        subprocess.run(["git", "add", "-A"], cwd=d, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m",
             f"{PROSE[name]['tagline']}\n\n"
             f"Extracted from a single repository at {sha[:12]}. One of nine: the planes of that "
             f"tree met only through shared types and through artifacts on disk, and making that "
             f"boundary physical is what lets a failure in one of them localise.\n\n"
             f"`acr` is a PEP 420 namespace package across all nine; no import statement changed."],
            cwd=d, check=True)
        n = len(list(d.rglob("*")))
        print(f"  {name:<22}{n:>6} files  deps={len(REPOS[name]['deps'])} sibling(s)")

        if args.push:
            vis = "--public" if args.public else "--private"
            r = subprocess.run(["gh", "repo", "create", name, vis, "--source=.",
                                "--description", PROSE[name]["tagline"], "--push"],
                               cwd=d, capture_output=True, text=True)
            print(f"      {'pushed' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip()[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
