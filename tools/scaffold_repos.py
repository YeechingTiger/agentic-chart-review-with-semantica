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
    "acr-contract": {
        "tagline": "A declarative extraction contract, and the record a run of it must leave.",
        "what": """A **task contract** is a YAML file that states a question, the numbered decision
rules that may establish an answer, the conflict rules that order them when two apply, the evidence
rules that say what counts as support, and the closed set of outcomes a run is allowed to conclude.
This package loads one, validates it, freezes it to a content hash, and defines the shape of the
answer and the trace that come back.

Nothing here runs an agent or reads a document. It is the vocabulary two other programs need in
order to disagree about the same thing — which is why it is a separate distribution with a version
number, and the only one of the nine that has one.""",
        "generalises": """Any task where a human-authored rule set has to be executed against
documents and the result has to be defensible afterwards: benefits eligibility, contract review,
incident triage, regulatory conformance, systematic review screening. The word "chart" appears in
this package's history, not in its interfaces.""",
        "not": """- Does not call a model, open a document, or decide anything clinical.
- Does not know what an identifier looks like at your site; see `acr/core/site.py`, which asks.
- Does not validate that a contract is *correct* — only that it is complete and internally
  consistent. Correctness is a clinician's job and `acr-spec-authoring` is where they do it.""",
        "hard_won": """**Every enforced element needs a provenance record, or loading fails.**
`load_spec` raises `UnprovenancedElementError` when a rule the runtime will act on carries no note
of where it came from, and `StaleProvenanceError` when a record names an element that is no longer
enforced. The default provenance is `model_authored`, deliberately, and not the manual named at the
top of the file — because naming a standards body in a header is not evidence that the sentence
three hundred lines down came out of it, and the opposite default lets a reviewer approve a
fabricated rule believing a standards body wrote it.

**The outcome space is a property of the contract, not of the code.** A run may conclude a value, or
abstain because the evidence is insufficient, or abstain because the contract does not cover the
case, or fail. Those are four different things and collapsing them loses the only signal that
distinguishes "the record is silent" from "we never asked the right question".""",
    },
    "acr-corpus": {
        "tagline": "A synthetic document corpus with declared ground truth, and the generator that "
                   "makes it byte-identically.",
        "what": """Twenty-seven synthetic patient records, 8,154 documents, each chart carrying a
`_ground_truth.json` that states the answer AND why that answer is the answer. Six of them are
**held out**: each was designed from a clause of a contract that no other chart exercises, and from
no observed run result, so a method scored on them is not being scored on its own development set.

`generate_corpus.py` is deterministic and seeded per patient. Regenerating produces the same bytes,
which is what makes an edit to a held-out chart visible.""",
        "generalises": """The pattern generalises even where the documents do not: a corpus whose
traps are derived from the CONTRACT rather than from watching a system fail is the only kind you can
score a method on twice. Six of these charts exist because the other twenty-one had already been
used to design the methods being measured.""",
        "not": """- Contains no real data. Nothing here is derived from a person.
- Is not a benchmark leaderboard. There are twenty-seven charts; the held-out denominator is six,
  and printing it small is the point.""",
        "hard_won": """**This is a separate repository because a `.gitignore` could not freeze it.**
The six held-out charts were generated on 2026-08-03 into a tree whose ignore file carried
`*[0-9].txt`, written to catch iCloud conflict copies (`Name 2.txt`) but missing the space — and a
document filename is `Doc-Type_2013-10-27.txt`, which ends in a digit. Already-tracked files were
unaffected, so nothing looked wrong, and the six held-out charts went into version control with ONE
file each. 1,589 documents were never committed.

That inverts the one guarantee a held-out set exists for. A tag on a repository is a freeze; an
ignore pattern is a hope.""",
    },
    "acr-chart-review": {
        "tagline": "Run one declarative contract over one document corpus, and leave a record that "
                   "distinguishes a correct answer from a lucky one.",
        "what": """The agent. It receives a contract, a corpus of a few hundred documents for one
subject, and prose method cards assembled by slot; it gets seven tools — list, summarise-by-type,
search, read, read-batch, record-evidence, submit — and it leaves a JSONL trace of every call plus a
JSON manifest stamped with enough identity that two runs are comparable: the contract's hash, the
corpus's content hash, the code SHA, the resolved arm.

The loop itself is off-the-shelf (deepagents / LangGraph). What is here is the part above it: the
tools, the typed state, the gate on the submitted answer, the coverage plan that governs what may be
opened, and the record.""",
        "generalises": """Substitute the corpus and the contract and nothing here knows the
difference. The three assumptions are: the documents are text with a type and a date, the question
is answerable from them, and somebody will later need to know WHY the answer was what it was.""",
        "not": """- Does not score itself. Scoring is `acr-eval`, deliberately in another
  distribution, because a runtime that can see the answer key is a runtime that will use it.
- Does not edit its own prompts. That is `acr-improvement`.
- Does not decide when a contract is wrong. It records that it could not answer and says which
  clause was silent.""",
        "hard_won": """**Most checks are advisory, and that was measured rather than chosen.** Five
deterministic content checks were removed after destroying 58 correct values against 21 helps. The
coverage gate went advisory after ~150 rejections of which 27 refused the reference value's exact
tuple. A thread refusal went advisory at 28% reference-destroying. The pattern held every time: a
rule that is right about the text is wrong about the answer more often than it is right.

**Of eleven wrong answers in the last valid batch, ZERO were retrieval failures.** `NEVER_LOOKED 0`,
`READ_NOT_CITED 0`. The agent opened the document carrying the answer every single time and got the
reading wrong. If you are about to invest in better search, measure this first — the tool for it
ships in `acr-harness`.""",
    },
    "acr-eval": {
        "tagline": "Score, judge, attribute and audit completed runs — from the record alone.",
        "what": """Four things that must not be one thing:

- **Score** a run against an answer key, deterministically, with detectors for the behaviours a
  score cannot see (zero documents read, a search that cannot fail, a rejection loop).
- **Judge** what no rule can score, fenced: a judged number is an OPINION and is refused wherever a
  deterministic evaluator exists.
- **Attribute** a wrong answer to a cause, by an agent that has never been shown the key.
- **Audit** the trajectory truth-blind: did this run touch a subject it was not reviewing, did an
  artifact leave its boundary, does a trace we already wrote to disk carry an identifier.

Everything reads finished manifests and traces. Nothing re-runs anything.""",
        "generalises": """Any agent that leaves a structured trace can be scored, judged, attributed
and audited by this package; the reader is schema-tolerant across the drift of its own history.
`RunRecord` is the interesting part — it pairs a manifest with its trace and every accessor on it is
a scar from a field that moved.""",
        "not": """- Never rewrites an answer. Three independent guards enforce that, because the
  first version of this plane did.
- Never claims a conclusion above its truth mode. GOLD, REGISTRY_REFERENCE and BLIND are a
  CEILING on what may be said, verified against the recorded runs.
- Does not fix anything. Routing a finding to an owner is `acr-improvement`.""",
        "hard_won": """**This plane's own accuracy has never been measured, and the code says so.**
`meta_evaluate_attributions` requires 30 adjudicated cases and a macro-F1 of 0.80 and has never
run; there are 2 attribution records on disk. Treat its output as a hypothesis until that number
exists. An evaluation plane nobody has evaluated is a plane with an unknown error rate, and the
honest thing is to print that rather than to imply otherwise.""",
    },
    "acr-experience": {
        "tagline": "Turn a labelled development set into retrieval assets an agent can be GIVEN.",
        "what": """Read every document of a development set once, cheaply, against ONE requirement.
Price each candidate term by what it actually retrieves. Write a retrieval plan. Certify it on a
held-out test set before anything at scale is allowed to use it.

The output is an INPUT to a run — a prior, handed to the agent in its own prompt slot. That is the
distinction from `acr-improvement`, which changes the system itself.""",
        "generalises": """Any retrieval task where the vocabulary of the question and the vocabulary
of the corpus are different, which is most of them. The full scan is the expensive honest baseline
that tells you whether an agent is earning anything over a query.""",
        "not": """- Does not touch the runtime. It writes assets; the runtime chooses to load them.
- Does not certify on the set it developed on. That refusal is the whole point of the two-set
  split, and `answer_leak.py` is the guard that a derived term is not the answer it was derived
  from.""",
        "hard_won": """**A term derived from an answer is not evidence that the term works.** The
guard exists because the failure is invisible: a keyword list derived from labelled data will score
beautifully on that data and add nothing anywhere else.

**And a caution about scope**: two of the six contracts this was built against have since had their
keyword lists and strata REMOVED on purpose, because measurement showed the agent did better
choosing its own terms. This plane refuses at the door rather than inventing assets for a contract
shape that declares none — which is correct, and also means it may be solving a problem your task
does not have. Measure retrieval reachability first.""",
    },
    "acr-improvement": {
        "tagline": "Reflective optimisation of the text an agent reads, routed from classified "
                   "failures and never applied unvalidated.",
        "what": """Every text parameter an agent reads is a parameter: the system prompt, the method
cards, the clauses of the contract itself. This package takes classified failures, routes each one to
the parameter that could have caused it, proposes an edit, and requires paired validation before the
edit may stand.

`BehaviorSignature` reduces a run to what it answered, what it cited, which rules it claimed and how
it got there, hashed — so "these two runs behaved the same" is a comparison and not an
impression.""",
        "generalises": """Any prompted system whose behaviour you are trying to move deliberately.
The routing is the substance: without it an optimiser reverse-engineers a story from the outcome and
starts confidently rewriting rules that were never at fault.""",
        "not": """- Never applies an edit. It proposes, and validation is a separate decision.
- Never edits a clinical rule on its own authority. A semantic change requires gold AND human
  adjudication; a REGISTRY_REFERENCE truth mode can only produce a question for a clinician.
- Does not derive retrieval assets. That is `acr-experience`, and the split is between changing
  the system and giving it an input.""",
        "hard_won": """**Refuse a case id that looks like a real identifier, at the door.** Routing
inputs carry case ids and the routed artifacts are written where a human will read them, so
`FailureCase` raises rather than pseudonymising quietly. The shape of an identifier is deployment
configuration (`acr-contract`'s `site.py`) and there is no default: three were tried and each was
measured wrong somewhere nobody had looked.""",
    },
    "acr-spec-authoring": {
        "tagline": "An arbitrary question becomes a declarative contract — checked for completeness, "
                   "and put in front of the person who owns its decisions.",
        "what": """Three things:

- **Intake**: an arbitrary question routes to a contract that can answer it, or to an explicit
  statement of what is missing. Never to a guess.
- **Lint**: eleven formal completeness checks in four tiers that cost four different things to run
  and mean four different things when they pass. A single PASS over all four is the sentence this
  tool exists to make unsayable.
- **Review**: render a contract as a document a domain expert can read in ten minutes and mark up,
  and record — with a content hash — that a named person approved one element as it was worded that
  day. The next render reports the approval as withdrawn the moment the wording changes.""",
        "generalises": """Wherever the person who owns the rules cannot read the file the rules live
in. The review renderer is the only part of this whole system whose user is not an engineer, and
that is a gap most rule-executing systems have and do not name.""",
        "not": """- Does not decide whether a rule is clinically right. It makes the rule legible and
  records who said it was.
- Does not run anything. A contract that lints clean is not a contract that works.""",
        "hard_won": """**Assent to a sentence is not assent to whatever that sentence is edited
into.** Sign-off records carry the element's content hash for exactly that reason.

**Five of the eleven lint checks produce zero findings over all six shipped contracts.** They are
regression guards in a passing state, not dead code — but do not read a clean lint as coverage. The
14 tier-1 failures the shipped contracts DO produce are real and unfixed.""",
    },
    "acr-concordance": {
        "tagline": "Given extracted variables, decide by rule whether a case conforms to a "
                   "guideline — and when it does not, which cause is standing.",
        "what": """A rule engine, with no model anywhere in its import closure. It reads variables
somebody else extracted, evaluates a guideline over them, and returns conformance or
non-conformance. Then, for a non-conforming case, it narrows to which of four causes survives:
the care itself, the documentation, the extraction, or a justified exception.

Everything after extraction is deterministic and replays from a file.""",
        "generalises": """Any conformance question over extracted structured data — clinical
guidelines, regulatory checklists, contractual obligations, eligibility rules. The dependency
direction is the useful part: it can tell you which variables a rule reads, and what breaks if one
of them moves.""",
        "not": """- Reaches no model. Not a default — a property, and the import closure is the
  proof.
- Does not extract anything. It consumes an extract.
- Does not collapse the four causes into one number. Two of them are about the patient and two are
  about the pipeline, and a single "non-concordance rate" hides which.""",
        "hard_won": """**Four causes, and they must not become one number.** A case can be
non-conforming because the care was wrong, because the care was right and the note is silent,
because the extraction missed it, or because an exception applies and is documented. Report them
apart or the metric will be optimised by improving extraction and nothing else.""",
    },
    "acr-harness": {
        "tagline": "Compare arms, not answers — and refuse the comparisons that would not mean "
                   "anything.",
        "what": """An experiment ladder. Each arm differs from the baseline in exactly one thing —
one method card, one runtime profile, one prior — and the harness runs the arms over a corpus,
records identity for every run, and analyses the results.

It also holds the refusals, which is most of its value:

- Refuses a comparison across mixed contract hashes, because that is the one axis under measurement.
- Refuses to fold a chart that INFORMED a method's design into a headline number.
- Refuses a headline over a batch where every chart is informed.
- Refuses to price a model it has no rate for, rather than reporting zero.""",
        "generalises": """Any A/B over a prompted system. The discipline is the transferable part:
freeze the protocol and register the predictions before the first call, report strata apart, and let
the analyser refuse rather than footnote.""",
        "not": """- Does not score an answer. That is `acr-eval`; this asks whether an
  INTERVENTION earned its place.
- Does not average seeds into one number without saying so. An effect whose ranking flips between
  seeds is noise wearing a result's clothes.""",
        "hard_won": """**A footnote loses; a refusal holds.** The warning that informed charts must
not enter a headline number was written as a footnote twice in this project's history and lost both
times. It is now arithmetic that stops.

**The protocol goes in before the first model call.** `docs/POLICY_LADDER_PROTOCOL.md` is the
example, including a prediction registered AGAINST the thing being tested — because a prediction
written afterwards is a description.""",
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
