"""`site.py`'s addresses must resolve from any working directory.

`acr` is an installed console script (`pyproject.toml` `[project.scripts]`), so it runs wherever the
operator happens to be. Every path in `site.py` goes through a parent search or `$HOME` — except one:

    PRICES = Path(os.getenv("ACR_PRICES", "assets/pricing/prices.json"))

cwd-relative. From `/tmp` the table does not exist, `Spend.usd` is `None` for EVERY model, and
`agent.py`'s `if spend.exceeded()` never trips — so `--max-usd`, which `acr label scan` makes
REQUIRED precisely so nobody spends without saying how much, silently enforces nothing.

Nothing published is wrong (`spend.report()` records the ceiling as unenforced and the cost column
reads `None`), but a ceiling that depends on your shell's cwd is not a ceiling.

`LOCAL_ROOT` is the other half: it had ZERO readers anywhere while its docstring named
`LocalArtifactStore` as its consumer, and the store reads `ACR_LOCAL_ARTIFACT_ROOT`. Two env vars
for one address, one of them dead, and `docs/NEW_TASK_NEW_DATA.md` pasted the dead one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _from_elsewhere(snippet: str) -> str:
    """Run a snippet with cwd=/ so a cwd-relative default cannot accidentally resolve."""
    out = subprocess.run([sys.executable, "-c", snippet], cwd="/", capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=False)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_the_price_table_resolves_from_another_directory():
    assert _from_elsewhere(
        "from acr.core import site; print(site.PRICES.is_file())") == "True"


def test_a_priced_model_is_still_priced_from_another_directory():
    """The consequence that matters: a ceiling that does not bind is not a ceiling."""
    got = _from_elsewhere(
        "from acr.core.spend import Spend;"
        "s = Spend(max_usd=0.01, model='openai/gpt-5.6-luna', prompt=10_000_000);"
        # `exceeded()` returns the MESSAGE, not a bool — `bool()` is what `agent.py` branches on.
        "print(s.usd is not None, bool(s.exceeded()))")
    assert got == "True True", got


def test_an_explicit_override_still_wins():
    assert _from_elsewhere(
        "import os; os.environ['ACR_PRICES'] = '/nope/prices.json';"
        "from acr.core import site; print(site.PRICES)") == "/nope/prices.json"


def test_there_is_one_local_root_env_var():
    """`ACR_LOCAL_ROOT` had no readers; the boundary is enforced against
    `ACR_LOCAL_ARTIFACT_ROOT`. Two names for one address is how a documented command hits a
    refusal — `docs/NEW_TASK_NEW_DATA.md` pasted `--local-root "$ACR_LOCAL_ROOT"`."""
    from acr.core import local_artifacts, site
    assert not hasattr(site, "LOCAL_ROOT"), (
        "site.LOCAL_ROOT is back. It has no readers; the store reads "
        f"{local_artifacts.LOCAL_ROOT_ENV}.")


def test_the_surviving_name_is_the_one_the_store_enforces():
    from acr.core import local_artifacts
    assert local_artifacts.LOCAL_ROOT_ENV == "ACR_LOCAL_ARTIFACT_ROOT"
