"""The end-to-end proof's free half, run in CI so the script cannot rot between real runs.

`tools/prove_end_to_end.py` is the answer to a specific failure: on 2026-08-06 the chain was walked
by hand and two steps were broken in ways 2188 passing tests could not see — a harness-profile key
guessed against a scripted fake, and an attribution budget too small to reach a verdict. Unit tests
drive fakes; only a real run falsifies a claim about a provider.

But a proof script that is only ever run by hand rots exactly as quietly as the prose it replaced.
So its FREE half — corpus load, contract lint, answer key — runs here, every suite, no model and no
money. That does not prove the paid half works; it proves the script still starts, still finds the
commands it drives, and still reports a step it could not complete as a failure rather than as
silence.

The paid half is deliberately not here. A test that spends money on every `pytest` run is a test
somebody disables.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/prove_end_to_end.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_the_free_half_of_the_chain_still_passes():
    """Corpus, contract and answer key, with no provider reachable."""
    r = _run("--dry-run")
    assert r.returncode == 0, f"the free chain failed:\n{r.stdout}\n{r.stderr}"
    for stage in ("check-corpus", "spec lint", "answer key"):
        assert f"PASS  {stage}" in r.stdout, f"{stage!r} did not pass:\n{r.stdout}"


def test_a_step_that_produced_nothing_is_reported_as_a_failure():
    """THE PROPERTY THE SCRIPT EXISTS FOR, asserted against the script itself.

    Every step asserts on an artifact rather than on an exit code, because a command that exits 0
    having written nothing is this repository's recurring failure — an inert check reads exactly
    like a satisfied one. The script shipped with that very bug: `all([])` is `True`, so the
    open-gap step passed over zero manifests on its first real run.

    Pointed at a corpus with no patients, the answer-key step must FAIL rather than report a clean
    zero.
    """
    import os
    env = {**os.environ, "ACR_CORPUS": str(ROOT / "tests")}   # a real directory, no patients in it
    r = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
    assert "FAIL" in r.stdout, (
        "a chain over an empty corpus reported no failing step; a proof that cannot fail is the "
        f"thing this script was written to replace:\n{r.stdout}")
    assert r.returncode == 1, "a failing step must set a non-zero exit code for CI to see"
