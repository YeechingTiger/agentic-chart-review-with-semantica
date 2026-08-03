"""Cut this repository into the nine it should have been, driven by a table rather than by hand.

    python tools/split_repos.py --out ../acr-split --dry-run
    python tools/split_repos.py --out ../acr-split

WHY A SCRIPT AND NOT NINE `git mv` SESSIONS. Every manual step in this split is a chance to drop a
file, and a dropped file does not announce itself: the losing repository still imports the name from
the *installed* sibling during development and only fails on a clean machine. The table below is the
whole decision, it is diffable, and `--dry-run` prints the assignment so a reader can argue with it
before anything moves.

THE MECHANISM IS PEP 420. `src/acr/__init__.py` was deleted first (24 re-exports, zero consumers),
which makes `acr` an implicit namespace package. So `src/acr/contract/` shipping from one
distribution and `src/acr/review/` from another both install into `acr.*`, and NOT ONE IMPORT
STATEMENT CHANGES across 26,000 lines. The alternative — renaming each plane to its own top-level
package — would have rewritten every import in every repo, which is a large diff whose failures are
silent typos.

HISTORY IS NOT CARRIED, deliberately. `git subtree split` can only reach back to `c0c8948`, the
commit that reorganised seventeen root directories into seven, so per-plane history is 1–21 commits
and blame would point at a move rather than at a decision. Each new repository therefore starts with
one commit that names the source SHA, and the original repository stays as the archive. The cost is
smaller than it looks: this tree's history lives mostly in its docstrings, and those travel with the
files.

WHAT EACH REPOSITORY OWES ITS READER, and why the READMEs are written rather than generated: every
one of these is meant to be usable on a task that is not cancer registry abstraction. A README that
says "extracts date_of_initial_diagnosis" describes a demo; one that says "runs a declarative
contract over a document corpus and leaves a record you can audit" describes a tool. The generated
part is the file inventory and the dependency block; the prose is hand-written per repo.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

from acr.core import site

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: `contract/` holds four modules that no runtime path touches — they belong to the planes that use
#: them, and leaving them in the shared layer is what made `tests/test_layering.py`'s claim that the
#: working planes "meet only through the shared layers" partly fictional. 2,256 code lines that
#: layer 2 never imports. The split is where that gets fixed rather than restated.
CONTRACT_SHARED = ("spec", "outcomes", "answer_contract", "answer_checks", "trace", "code_tables",
                   "site_mapping", "strata", "case_requirements", "skills")

#: name -> (one-line purpose, source paths, test files, runtime deps on siblings)
#: A source path ending in `/` takes the whole directory; otherwise it is one file.
REPOS: dict[str, dict] = {
    "acr-contract": {
        "purpose": "The shared vocabulary: what a task contract IS, what an answer may be, and what "
                   "a run must leave behind.",
        "src": [f"src/acr/contract/{m}.py" for m in CONTRACT_SHARED]
               + ["src/acr/core/", "src/acr/chartstore/"],
        "assets": ["assets/contracts/", "assets/codes/", "assets/pricing/"],
        "deps": [],
    },
    "acr-corpus": {
        "purpose": "A synthetic document corpus with declared ground truth, and the deterministic "
                   "generator that produces it.",
        "src": ["tools/generate_corpus.py"],
        "data": ["corpus/"],
        "deps": ["acr-contract"],
    },
    "acr-chart-review": {
        "purpose": "The agent: run one declarative contract over one document corpus, with seven "
                   "read/record/submit tools and an immutable record of what it did.",
        "src": ["src/acr/review/", "src/acr/commands/cli_chart.py"],
        "assets": ["assets/skills/", "assets/specs/", "assets/module_catalog/runtime_policies/"],
        #: minus the cards other consumers own — applied in `collect`
        "deps": ["acr-contract"],
    },
    "acr-eval": {
        "purpose": "Score, judge, attribute and audit COMPLETED agent runs, from the record alone "
                   "and never by re-running them.",
        "src": ["src/acr/evaluation/evals.py", "src/acr/evaluation/judge.py",
                "src/acr/evaluation/evaluation_pipeline.py",
                "src/acr/evaluation/evaluation_modules.py",
                "src/acr/evaluation/evidence_chain.py",
                "src/acr/diagnosis/", "src/acr/audit/",
                "src/acr/commands/cli_eval.py", "src/acr/commands/cli_evaluation.py",
                "src/acr/commands/cli_judge.py", "src/acr/commands/cli_attribute.py",
                "src/acr/commands/cli_audit.py", "src/acr/commands/cli_signal.py"],
        "assets": ["assets/module_catalog/audit_rules/", "assets/module_catalog/evaluators/",
                   "assets/evaluators/", "assets/certification_catalog/",
                   "assets/pipeline_catalog/"],
        "deps": ["acr-contract"],
    },
    "acr-experience": {
        "purpose": "Turn a labelled development set into retrieval assets an agent can be GIVEN: "
                   "read every document once, price the words, certify on held-out data.",
        "src": ["src/acr/improvement/labelling.py", "src/acr/improvement/derive.py",
                "src/acr/improvement/assetdev.py", "src/acr/improvement/answer_leak.py",
                "src/acr/commands/cli_label.py"],
        "assets": ["assets/experience/"],
        "deps": ["acr-contract"],
    },
    "acr-improvement": {
        "purpose": "Reflective optimisation of the TEXT an agent reads — prompts, cards, "
                   "specification clauses — routed from classified failures and never applied "
                   "unvalidated.",
        "src": ["src/acr/improvement/refine.py", "src/acr/contract/spec_repair.py",
                "src/acr/commands/cli_refine.py", "src/acr/commands/cli_repair.py"],
        "deps": ["acr-contract"],
    },
    "acr-spec-authoring": {
        "purpose": "An arbitrary question becomes a declarative contract, checked for completeness "
                   "and put in front of the person who owns its decisions.",
        "src": ["src/acr/authoring/", "src/acr/usecase/",
                "src/acr/contract/registry_catalog.py",
                "src/acr/commands/cli_spec.py", "src/acr/commands/cli_plan.py",
                "src/acr/commands/cli_site_mapping.py"],
        "deps": ["acr-contract"],
    },
    "acr-concordance": {
        "purpose": "Given extracted variables, decide by RULE whether a case conforms to a "
                   "guideline — and when it does not, which of four causes is standing.",
        "src": ["src/acr/contract/concordance.py", "src/acr/contract/deps.py",
                "src/acr/evaluation/explain.py",
                "src/acr/commands/cli_pipeline.py", "src/acr/commands/cli_gold.py"],
        "assets": ["assets/guidelines/", "assets/usecase/"],
        "deps": ["acr-contract"],
    },
    "acr-harness": {
        "purpose": "Compare ARMS, not answers: run a controlled ladder over a corpus and refuse the "
                   "comparisons that would not mean anything.",
        "src": [],  # filled from tools/ below, minus the ones that belong elsewhere
        "docs": ["docs/"],
        "deps": ["acr-contract", "acr-chart-review", "acr-eval"],
    },
}

#: Tests that import no `acr` module at all, routed by what they are ABOUT. `test_adversarial_corpus`
#: reads chart files and asserts the traps are where the ground truth says: it belongs with the data,
#: and an import graph cannot see that because it imports nothing.
TESTS_BY_SUBJECT = {
    "test_adversarial_corpus.py": "acr-corpus",
    #: A card's test belongs with the card. `assets/skills/` split four ways, so three of these
    #: were testing an asset that had moved out from under them — which is the same class of bug as
    #: the card assignment itself, one level up.
    "test_store_to_spec_skill.py": "acr-spec-authoring",
    "test_guideline_to_rules_skill.py": "acr-concordance",
    #: Validates EVERY card in the tree against the slot contract. Its subject is the whole card
    #: set, which now lives in four repositories, so it can only run where all four are present.
    "test_skills_load.py": "(composer)",
    "test_eval_skill_fence.py": "acr-eval",
    #: Its subject is the CRC use-case bundle under `assets/usecase/`, which went with the
    #: guideline plane. An import graph put it in the shared layer because it imports nothing.
    "test_crc_full_normalization.py": "acr-concordance",
    #: Loads `tools/run_ladder.py` by path and validates every arm. The driver is the harness's.
    "test_run_ladder_arms.py": "acr-harness",
}

#: WHO OWNS WHICH METHOD CARD. `assets/skills/` held cards for FOUR different consumers in one
#: directory, and the split is what made that visible: the five `slot: eval` cards are read by the
#: evaluation agent, not by the chart-review agent, and `acr-eval`'s tests fail at once when they
#: ship with the wrong repository. Two `slot: task` cards are about AUTHORING a contract rather than
#: answering one, and `non-concordance-triage` explains a guideline verdict. Assigning by slot alone
#: would have put all four in the wrong place.
CARD_OWNERS = {
    "acr-eval": ("eval-cluster-failures", "eval-contrast-traces", "eval-key-challenge",
                 "eval-missed-evidence", "eval-overconfidence"),
    "acr-spec-authoring": ("store-to-spec",),
    #: `guideline-to-rules` has references and NO `SKILL.md`: the agent authoring it was killed
    #: mid-work by an org spend limit, and the half-written card was committed. It travels with the
    #: engine it teaches, and `tests/test_guideline_to_rules_skill.py` goes with it.
    "acr-concordance": ("crc-guideline-registry-authoring", "non-concordance-triage",
                        "guideline-to-rules"),
}

#: The two scripts that perform the split itself. They belong to the archive repository, not to the
#: harness: after the split they have no subject, and `scaffold_repos` imports `split_repos`, so a
#: harness that took one and not the other would not import.
SPLIT_TOOLS = ("split_repos.py", "scaffold_repos.py")

#: Tools that belong to a specific repo rather than to the harness.
TOOL_HOMES = {
    "generate_corpus.py": "acr-corpus",
    "render_chain.py": "acr-eval",
}

#: Stays in the original repository, which becomes the COMPOSER: the `acr` console script that
#: mounts every sibling's command group, the cross-plane tests, and the documents that are about the
#: whole system rather than about one plane. `cli.py` imports from all ten planes and is the only
#: fan-out hub in the tree; there is nowhere else it could live.
COMPOSER_KEEPS = ("src/acr/commands/cli.py",)

#: Tests whose imports span more than one working plane, or which are ABOUT the composition. They
#: stay in the original repository, which becomes the composer. Computed, not guessed — see
#: `assign_tests`.
COMPOSER_TESTS_NOTE = "cross-plane by import, or about the composition itself"


def plane_of(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) >= 2 and parts[0] == "acr" else None


def repo_of_source(path: str) -> str | None:
    for name, spec in REPOS.items():
        for p in spec.get("src", []):
            if p.endswith("/") and path.startswith(p):
                return name
            if path == p:
                return name
    return None


def assign_tests() -> dict[str, list[str]]:
    """Route each test file by the planes it imports; anything spanning two working planes stays."""
    import ast
    out: dict[str, list[str]] = {k: [] for k in REPOS}
    out["(composer)"] = []
    for t in sorted((ROOT / "tests").glob("*.py")):
        if t.name == "conftest.py":
            continue
        try:
            tree = ast.parse(t.read_text(encoding="utf-8"))
        except SyntaxError:
            out["(composer)"].append(t.name)
            continue
        homes = set()
        for n in ast.walk(tree):
            # `from acr.improvement import refine` names the PACKAGE in `n.module` and the module
            # in the alias. Resolving only `n.module` sent three tests to the shared layer, because
            # `src/acr/improvement.py` matches nothing and the fallback was contract.
            mods = ([n.module] + [f"{n.module}.{a.name}" for a in n.names]
                    if isinstance(n, ast.ImportFrom) and n.module and not n.level
                    else [a.name for a in n.names] if isinstance(n, ast.Import) else [])
            for m in mods:
                pl = plane_of(m)
                if not pl:
                    continue
                # Resolve to a repo through the module path, so a `contract.*` module that moved
                # out of the shared layer routes to the repo that took it.
                mod_path = "src/" + m.replace(".", "/") + ".py"
                if m.startswith("acr.commands.cli") and m.count(".") == 2:
                    # `acr.commands.cli` is the composer: it mounts every sibling's group and
                    # imports from all ten planes. A test that drives it is a test of the
                    # composition, whatever else it touches.
                    homes.update({"(composer)", "_"})
                    continue
                r = repo_of_source(mod_path) or repo_of_source(f"src/acr/{pl}/")
                if r and r != "acr-contract":
                    homes.add(r)
        if t.name in TESTS_BY_SUBJECT:
            out[TESTS_BY_SUBJECT[t.name]].append(t.name)
            continue
        out[next(iter(homes)) if len(homes) == 1 else "(composer)"
            if homes else "acr-contract"].append(t.name)
    return out


def collect(spec: dict, name: str | None = None) -> list[str]:
    """Files this repo takes. Cards owned by another consumer are subtracted, and added back to the
    owner, so `assets/skills/` splits four ways instead of travelling whole with the agent."""
    others = {c for owner, cards in CARD_OWNERS.items() if owner != name for c in cards}
    mine = CARD_OWNERS.get(name or "", ())
    files: list[str] = []
    for card in mine:
        d = site.skills_root() / card
        if d.is_dir():
            files += [str(f.relative_to(ROOT)) for f in d.rglob("*") if f.is_file()]
    for key in ("src", "assets", "data", "docs"):
        for p in spec.get(key, []):
            base = ROOT / p
            if p.endswith("/"):
                if base.is_dir():
                    files += [str(f.relative_to(ROOT)) for f in base.rglob("*")
                              if f.is_file() and "__pycache__" not in f.parts]
            elif base.is_file():
                files.append(p)
    if others:
        files = [f for f in files
                 if not any(f.startswith(f"assets/skills/{c}/") for c in others)]
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory to create the repositories in")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # tools/ -> harness, minus the ones with a specific home
    REPOS["acr-harness"]["src"] = [
        f"tools/{f.name}" for f in sorted((ROOT / "tools").glob("*.py"))
        if TOOL_HOMES.get(f.name) is None and f.name not in SPLIT_TOOLS]

    tests = assign_tests()
    out = pathlib.Path(args.out).expanduser().resolve()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()

    print(f"source {sha[:12]}   ->   {out}\n")
    total = 0
    for name, spec in REPOS.items():
        files = collect(spec, name)
        total += len(files)
        print(f"{name:<22}{len(files):>5} files  {len(tests[name]):>3} tests  "
              f"deps={','.join(spec['deps']) or '-'}")
    print(f"{'(composer, stays)':<22}{'':>5}         {len(tests['(composer)']):>3} tests"
          f"   {COMPOSER_TESTS_NOTE}")
    print(f"\n{total} files assigned")

    unassigned = []
    for f in sorted(ROOT.glob("src/acr/**/*.py")):
        rel = str(f.relative_to(ROOT))
        if "__pycache__" in rel:
            continue
        if rel in COMPOSER_KEEPS:
            continue
        if not any(rel in collect(sp, n) for n, sp in REPOS.items()):
            unassigned.append(rel)
    if unassigned:
        print(f"\n!! {len(unassigned)} source files assigned to NO repo:")
        for u in unassigned:
            print(f"     {u}")
        if not args.dry_run:
            print("\nRefusing to write: an unassigned module is a module that silently disappears.")
            return 1
    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for name, spec in REPOS.items():
        dest = out / name
        if dest.exists():
            shutil.rmtree(dest)
        for rel in collect(spec, name):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        # Test-local helper modules travel with any repo whose tests import them. A bare
        # `from hooks_harness import ...` is not a third-party dependency, and a repo that takes the
        # test without the helper fails at collection rather than at a useful place.
        helpers = [h.name for h in (ROOT / "tests").glob("*.py")
                   if h.name != "conftest.py" and not h.name.startswith("test_")
                   and any(h.stem in (ROOT / "tests" / t).read_text(encoding="utf-8")
                           for t in tests[name])]
        for t in tests[name] + ["conftest.py"] + helpers:
            target = dest / "tests" / t
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "tests" / t, target)
        print(f"  wrote {name}")
    print(f"\n{len(REPOS)} repositories staged in {out}. "
          f"pyproject/README/git are written by the per-repo step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
