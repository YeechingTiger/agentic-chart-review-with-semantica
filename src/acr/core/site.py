"""Everything about a DEPLOYMENT that is not the code: where things live, and what an identifier
at this site looks like.

WHY THIS MODULE EXISTS. Until 2026-08-03 the answers were literals scattered across seven modules
— an institutional filesystem prefix in `core/usage_telemetry.py`, `core/spend.py`,
`improvement/labelling.py` and `improvement/derive.py`, and the same person-id regex compiled
separately in `contract/spec_repair.py`, `evaluation/evals.py`, `improvement/refine.py` and
`audit/audit_loop.py`. Two things were wrong with that, and they are one thing seen from two sides:

  * IT CANNOT BE PUBLISHED. A regex for one institution's record numbers is not itself PHI, but it
    discloses their SHAPE, and a fixed prefix with a known length is most of what a guesser needs.
    An absolute path discloses a filesystem layout and a project name. Neither belongs in a
    repository anyone else can read.
  * IT CANNOT BE REUSED. A package that hardcodes one site's addresses is a package with one user.

WHAT IS NOT CONFIGURABLE, and must not become so: that a person id is REFUSED where one should
never appear, and that patient-derived artifacts live outside the Git worktree. Those are the
mechanisms. This module holds their addresses, not their existence.

## The person-id pattern has NO DEFAULT, and that conclusion cost three attempts

`ACR_PERSON_ID_PATTERN` is unset until a deployment sets it, and while it is unset every guard that
depends on it is INERT and says so. That is not laziness. Three defaults were tried and each was
measured wrong somewhere nobody had looked:

  * The literal that was here: a fixed four-digit prefix and twelve more digits. Correct for one
    site, publishable by nobody. `audit/audit_loop.py` had built its copy by concatenating the
    prefix onto the digit class so that `tests/test_no_phi_in_tree.py`'s byte scan would not flag
    the file holding it. A guard you have to hide from your own guard is the wrong shape.
  * Ten or more digits with word boundaries both sides. LEAKS. A real instance id joins an
    identifier and a field id with a double underscore, `_` is a word character, the trailing
    boundary therefore fails, and the identifier passes through unmasked. Caught by
    `tests/test_evals.py::test_a_report_never_carries_a_real_person_id`, which exists because this
    already happened once.
  * The same, minus float mantissas, added after the unanchored form was found to match 3,934
    distinct digit runs across the recorded manifests, every one of them the tail of a serialised
    float. Zero false positives across 200 manifests, all of `src/` and all of `tests/`. Then 203
    across `assets/`, where content hashes begin with ten or more decimal digits often enough to
    matter. Every narrowing found a new collision one directory further out.

The two consumers want OPPOSITE error costs, which is why no single heuristic serves both, and why
there are now TWO settings rather than one:

  * `ACR_PERSON_ID_PATTERN` is what the RUNTIME refuses and masks. It must not false-NEGATIVE, or
    an identifier reaches a report a human will read. Err wide.
  * `ACR_PHI_SCAN_PATTERN` is what the pre-commit byte scan forbids anywhere in the tree. It must
    not false-POSITIVE, or it blocks every commit over a content hash that happens to start with
    ten digits. Err narrow, and make it specific to the shape this site actually issues.

Collapsing them into one setting is what produced the third failure above: a pattern wide enough to
mask safely flagged 203 asset hashes, and a pattern narrow enough to scan safely let an instance id
through unmasked. A library cannot know which digits are identifiers at a site it has never seen.
So it asks, twice.

FAIL-CLOSED, which is what makes an unset pattern safe rather than merely quiet: a deployment that
sets `ACR_REAL_CORPUS` has real data, and MUST also set `ACR_PERSON_ID_PATTERN`.
`require_person_id_pattern()` raises on that combination and
`tests/test_no_phi_in_tree.py::test_a_real_corpus_without_an_identifier_pattern_is_refused` pins
it. With no real corpus configured there is nothing for the pattern to protect, and inert guards
are the correct state.

DO NOT set it narrower than the identifiers actually in use. `evaluation/evals.mask_person_ids`
records what that costs: masking collapsed ten patients onto one `instance_id`, a ten-patient
before/after reported `0 regressions` while two instances had visibly left a good outcome, and no
test caught it because a synthetic `SYN0001` matches nothing and never collides.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path

# `LOCAL_ROOT` / `ACR_LOCAL_ROOT` was here and is DELETED. It had zero readers anywhere in the
# tree, while its own docstring named `LocalArtifactStore` as its consumer — and the store reads
# `ACR_LOCAL_ARTIFACT_ROOT` (`core/local_artifacts.LOCAL_ROOT_ENV`). Two env var names for one
# address, one of them dead, and `docs/NEW_TASK_NEW_DATA.md` pasted the dead one into a documented
# command: `--local-root "$ACR_LOCAL_ROOT"` expands to empty and hits the store's refusal.
# `tests/test_site_addresses_resolve_anywhere.py` keeps it gone.

#: Where the usage/cost callback writes. Outside the tree because it is shared with the LiteLLM
#: path's `sitecustomize` hook.
AUDIT_DIR = Path(os.getenv("ACR_AUDIT_DIR", str(Path.home() / ".acr" / "audit"))).expanduser()

def _prices_path() -> Path:
    """The model price table, resolved from a REPO ROOT rather than the working directory.

    This was `Path(os.getenv("ACR_PRICES", "assets/pricing/prices.json"))` — the one cwd-relative
    default in this module. `acr` is an installed console script, so from any directory but the
    checkout the table did not exist, `Spend.usd` was `None` for every model, and
    `agent.py`'s `if spend.exceeded()` never tripped: `--max-usd` enforced nothing. Nothing
    published was wrong — `spend.report()` records the ceiling as unenforced — but a ceiling that
    depends on the operator's shell is not a ceiling.

    Falls back to the literal relative path when there is no repo root (a packaged install with no
    checkout), because that is the state where there is genuinely no shipped table and the caller
    must set `ACR_PRICES`. `spend.report()` already says so in every manifest.
    """
    override = os.getenv("ACR_PRICES")
    if override:
        return Path(override).expanduser()
    rel = "assets/pricing/prices.json"
    try:
        from .repo_paths import repo_root
        return repo_root() / rel
    except Exception:                          # noqa: BLE001 — no checkout is a legitimate state
        return Path(rel)


#: The model price table. ONE table, because two price tables is two answers to "what did this
#: cost". Ships in-repo; a site with negotiated rates overrides via `ACR_PRICES`.
PRICES = _prices_path()

#: Where full-scan labellings live. Outside the tree: a labelling names documents.
LABELS_ROOT = Path(
    os.getenv("ACR_DEVLABELS_ROOT", str(Path.home() / ".acr" / "devlabels"))).expanduser()

#: Where the cached document bitmaps live, built once and read many times.
TERMCACHE_ROOT = Path(
    os.getenv("ACR_TERMCACHE_ROOT", str(Path.home() / ".acr" / "termcache"))).expanduser()

#: A file of `KEY=value` lines holding model credentials, read as DATA and never sourced as a shell
#: script — the module that reads it must not be able to execute what is in there. No default: a
#: deployment that has not said where its credentials are does not have any.
MODEL_ENV_FILE = os.getenv("ACR_MODEL_ENV_FILE") or None

#: The real, non-synthetic corpus, if this deployment has one. No default, and nothing in the
#: repository may assume it exists.
REAL_CORPUS_ROOT = os.getenv("ACR_REAL_CORPUS") or None

#: Where the pseudonym map lives. Outside the tree by construction: a map that travels with the
#: pseudonyms it protects is not a protection.
PSEUDONYM_MAP = Path(
    os.getenv("ACR_PSEUDONYM_MAP",
              str(Path.home() / ".acr" / "phi_pseudonym_map.json"))).expanduser()


# ----------------------------------------------------------------- shared fixture directories
#
# THE CORPUS, THE CONTRACTS AND THE METHOD CARDS live in one repository each, and every other
# repository reads them. Before the 2026-08-03 split they were all `<repo root>/corpus/patients`
# and `<repo root>/assets/specs`, hardcoded at about 130 call sites — which is correct in a
# monorepo and false everywhere else.
#
# THESE ARE FUNCTIONS, NOT CONSTANTS, and that is the whole design:
#
#   * A module constant is computed at import time, so it cannot see an environment a test sets
#     afterwards, and it cannot fail at a moment when the failure would make sense.
#   * The failure mode being fixed is a CONFUSING SYMPTOM. With the corpus missing, 24 tests
#     reported `UNKNOWN_PATIENT` — which reads as "this chart is not in the corpus" and sent me
#     looking at the corpus index. A resolver that raises `set ACR_CORPUS` costs one line and
#     saves that hour.
#
# Resolution order, first hit wins: the environment variable; then `<cwd>/…` and each parent, so a
# monorepo checkout needs no configuration; then a SIBLING CHECKOUT — `../acr-corpus/corpus/…` —
# because side-by-side clones are how these repositories are actually developed and requiring
# configuration for the common layout is how configuration gets ignored.

def _resolve(env_var: str, relative: str, sibling_repo: str, what: str) -> Path:
    override = os.getenv(env_var)
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{env_var}={override!r} does not exist ({what})")
        return p
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        if (base / relative).is_dir():
            return base / relative
        if (base.parent / sibling_repo / relative).is_dir():
            return base.parent / sibling_repo / relative
    raise FileNotFoundError(
        f"cannot find {what}. Looked for {relative!r} under {here} and its parents, and for a "
        f"sibling checkout of {sibling_repo!r}. Set {env_var} to the directory, or clone "
        f"{sibling_repo} beside this repository.")


#: Env var naming the Site Mapping JSON for this deployment's corpus. Read by surfaces that take
#: no command line, because a mapping belongs to a CORPUS and not to a request.
SITE_MAPPING_ENV = "ACR_SITE_MAPPING"


#: Env var holding the pseudonymisation secret. ONE NAME. `audit_loop._fingerprint` read
#: `ACR_PHI_FINGERPRINT_KEY` — which appeared in the whole tree exactly once, on that line — while
#: `evals`, the `audit-phi-in-trace` skill and the README used `ACR_PSEUDONYM_KEY`. A site that set
#: the documented name got `<redacted:no-local-key>` for every `acr audit run` finding, so a report
#: could not tell ONE identifier leaking forty times from FORTY leaking once, and the skill's
#: fingerprints over the same trace joined to nothing.
PSEUDONYM_KEY_ENV = "ACR_PSEUDONYM_KEY"

#: 12 hex characters, because two of the three call sites already used 12 and the `<person:…>`
#: tokens in recorded eval output are that length. Truncation length is part of the identifier: a
#: 16-character fingerprint and a 12-character one are not comparable by equality.
PSEUDONYM_DIGEST_CHARS = 12


def fingerprint(value: str) -> str:
    """A stable, keyed pseudonym for one identifier — the same one in every plane.

    Lives in `core` because `audit` and `evaluation` are sibling work planes and
    `tests/test_layering.py` forbids one importing the other. Two implementations of "what key
    fingerprints PHI" is how they came to disagree.

    UNKEYED IS REDACTED, NOT HASHED. An unkeyed digest of a small, structured identifier space —
    which a medical record number is — is a lookup table, not a protection.
    """
    key = os.environ.get(PSEUDONYM_KEY_ENV, "")
    if not key:
        return "<redacted:no-local-key>"
    return hmac.new(key.encode("utf-8"), str(value).encode("utf-8"),
                    hashlib.sha256).hexdigest()[:PSEUDONYM_DIGEST_CHARS]


def corpus_root() -> Path:
    """The document corpus. Ships in `acr-corpus`; `$ACR_CORPUS` overrides."""
    return _resolve("ACR_CORPUS", "corpus/patients", "acr-corpus", "the document corpus")


def specs_root() -> Path:
    """The task contracts. Ships in `acr-chart-review`; `$ACR_SPECS` overrides."""
    return _resolve("ACR_SPECS", "assets/specs", "acr-chart-review", "the task contracts")


def skills_root() -> Path:
    """The FIRST method-card directory. Prefer `skill_roots()` — see why below."""
    return skill_roots()[0]


def skill_roots() -> tuple[Path, ...]:
    """EVERY method-card directory, in search order.

    A tuple and not a path, because after the 2026-08-03 split the cards are distributed by
    CONSUMER rather than gathered in one place: the chart-review agent's task/policy/tactic/
    experience/general cards ship with the agent, the five `slot: eval` cards ship with the
    evaluation plane, `store-to-spec` with the contract authoring tools, and
    `non-concordance-triage` with the guideline engine. A single root was the monorepo's shape and
    resolving one made a card that had moved look like a card that never existed.

    Order: `$ACR_SKILLS` if set, then `assets/skills` under the cwd and each parent, then the same
    under every sibling directory of each of those parents. A card is looked up by NAME across all
    of them, which is the whole linkage — no repository has to know which sibling owns a card, only
    that the name means the same thing everywhere.
    """
    if os.getenv("ACR_SKILLS"):
        return (_resolve("ACR_SKILLS", "assets/skills", "acr-chart-review", "the method cards"),)
    roots: list[Path] = []
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        if (base / "assets" / "skills").is_dir():
            roots.append(base / "assets" / "skills")
        # Sibling checkouts, but NOT at the filesystem root: `/` holds entries that raise
        # `OSError: Invalid argument` on stat (`/.resolve` on macOS), and walking every top-level
        # directory looking for `assets/skills` is not a search, it is a scan of the machine.
        if base.parent != base and base.parent.parent != base.parent:
            try:
                siblings = sorted(base.parent.iterdir())
            except OSError:
                siblings = []
            for sib in siblings:
                d = sib / "assets" / "skills"
                try:
                    if d.is_dir() and d not in roots:
                        roots.append(d)
                except OSError:
                    continue
        if roots:
            break
    if not roots:
        raise FileNotFoundError(
            "cannot find the method cards. Looked for 'assets/skills' under "
            f"{here}, its parents, and their sibling directories. Set ACR_SKILLS, or clone "
            "acr-chart-review beside this repository.")
    return tuple(roots)

#: See the module docstring for the three defaults that were tried and what each one got wrong.
#: `None` until a deployment says what an identifier looks like here.
PERSON_ID_PATTERN = os.getenv("ACR_PERSON_ID_PATTERN") or None
PERSON_ID = re.compile(PERSON_ID_PATTERN) if PERSON_ID_PATTERN else None

#: What the pre-commit byte scan forbids in the tree. A DIFFERENT setting from the one above, and
#: the module docstring says why. Narrow: a false positive here blocks a commit.
PHI_SCAN_PATTERN = os.getenv("ACR_PHI_SCAN_PATTERN") or None
PHI_SCAN = re.compile(PHI_SCAN_PATTERN) if PHI_SCAN_PATTERN else None


def looks_like_a_person_id(value: object) -> bool:
    """True when `value` contains something shaped like a real identifier AT THIS SITE.

    One predicate rather than four separately compiled patterns, because the copies this replaced
    could drift and the direction they drift in is narrower, which is the direction that fails
    silently. False when no pattern is configured: with no real corpus there is nothing to guard,
    and `require_person_id_pattern` is what refuses the combination that would be unsafe.
    """
    return bool(PERSON_ID.search(str(value or ""))) if PERSON_ID else False


def require_person_id_pattern() -> None:
    """Refuse to proceed when a real corpus is configured and its identifier shape is not.

    The one place the two settings are linked. A deployment with `ACR_REAL_CORPUS` set is handling
    exactly the data the masking and the refusals exist for; leaving `ACR_PERSON_ID_PATTERN` unset
    there turns every one of them into a no-op, and a no-op guard reads exactly like a satisfied
    one.
    """
    missing = [name for name, value in (("ACR_PERSON_ID_PATTERN", PERSON_ID_PATTERN),
                                       ("ACR_PHI_SCAN_PATTERN", PHI_SCAN_PATTERN)) if not value]
    if REAL_CORPUS_ROOT and missing:
        raise RuntimeError(
            f"ACR_REAL_CORPUS is set but {' and '.join(missing)} is not. Every person-id refusal, "
            "every mask and the pre-commit scan are inert without them, and an inert guard is "
            "indistinguishable from a satisfied one. Set them for this site's identifier shape, or "
            "unset ACR_REAL_CORPUS.")
