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

import os
import re
from pathlib import Path

#: Root for anything derived from patient data that must not enter the worktree. A deployment
#: points this at storage it controls; the default is per-user and outside any checkout.
#: `core/local_artifacts.LocalArtifactStore` refuses a root that is relative or inside the worktree
#: regardless of what is set here — this variable chooses WHERE, not WHETHER.
LOCAL_ROOT = Path(os.getenv("ACR_LOCAL_ROOT", str(Path.home() / ".acr" / "local"))).expanduser()

#: Where the usage/cost callback writes. Outside the tree because it is shared with the LiteLLM
#: path's `sitecustomize` hook.
AUDIT_DIR = Path(os.getenv("ACR_AUDIT_DIR", str(Path.home() / ".acr" / "audit"))).expanduser()

#: The model price table. ONE table, because two price tables is two answers to "what did this
#: cost". Ships in-repo; a site with negotiated rates overrides.
PRICES = Path(os.getenv("ACR_PRICES", "assets/pricing/prices.json")).expanduser()

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
