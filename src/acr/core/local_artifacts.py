"""Hard storage boundary for patient-derived DEVELOP artifacts.

`.gitignore` is not a security boundary.  A JSON file is not ignored by the repository's
run-output rules, an ignored file can still be force-added, and a symlink can make a path that
looks external resolve back into the worktree.  Every command handling registry references,
case maps, chart-observable gold, traces, or attribution reports therefore comes through this
module before it reads or writes anything.

The store is deliberately a directory, not a database or dataset registry.  Its contents may
contain PHI even when case identifiers are pseudonymous; nothing here calls them de-identified
or shareable.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

LOCAL_ROOT_ENV = "ACR_LOCAL_ARTIFACT_ROOT"


class LocalArtifactError(ValueError):
    """A sensitive artifact escaped (or could escape) the declared local root."""


def _git_root() -> Path:
    try:
        raw = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
            check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalArtifactError(
            "cannot establish the Git worktree boundary; run inside the repository"
        ) from exc
    if not raw:
        raise LocalArtifactError("git rev-parse returned an empty worktree root")
    return Path(raw).resolve()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_run_artifact(value: str | Path, *, what: str = "run artifact") -> Path:
    """An existing RUN RECORD — a manifest or a trace — which lives inside the worktree by design.

    WHY THIS EXISTS. Every command that reads a completed run used `LocalArtifactStore.require_input`,
    which proves a path resolves UNDER the local root, and the local root is required to be outside
    the Git worktree. But `runs/` is inside the worktree — deliberately, gitignored wholesale — so the
    proof could never succeed and `acr audit run` could not be pointed at any of the 508 manifests on
    disk. The plane looked unexercised; it was unreachable. An adversarial review named this as the
    highest-value fix available, and running the chain end to end reproduced it independently.

    THE PROPERTY WORTH KEEPING IS NOT THE ONE THAT WAS CHECKED. The danger was never "reading a file
    inside the worktree". It was patient-derived material being COMMITTED. So this asserts the thing
    that actually matters: the file exists, and Git does not TRACK it. A tracked manifest is either
    synthetic (fine, and the caller may pass `allow_tracked=True` for a fixture) or a disclosure that
    has already happened, and reading it is not the moment to discover that — `tests/
    test_no_phi_in_tree.py::test_no_run_output_is_tracked` is what holds the property, and this is the
    same rule stated where a reader of one run will meet it.
    """
    raw = Path(value).expanduser()
    resolved = (raw if raw.is_absolute() else Path.cwd() / raw).resolve(strict=False)
    if not resolved.is_file():
        raise LocalArtifactError(f"{what} not found: {resolved}")
    if _is_git_tracked(resolved):
        raise LocalArtifactError(
            f"{what} is TRACKED by Git: {resolved}. Run output is never committed — `runs/` is "
            f"ignored wholesale — so a tracked run record is either a fixture (pass it directly) or "
            f"material that should not be in the tree at all.")
    return resolved


#: What a directory of run records is allowed to contain. A batch is named by its directory and
#: found by glob, so the glob IS the definition of the batch — widening it later silently changes
#: what every past comparison was over.
RUN_RECORD_GLOB = "*.manifest.json"


def require_run_tree(value: str | Path, *, what: str = "runs") -> Path:
    """A run record OR a directory of them, resolved for reading. The plural of the above.

    WHY A SECOND FUNCTION. `require_run_artifact` takes one file, and three planes take a batch:
    `attribute batch`, `eval score` and `repair` all accept "a manifest file or a directory". All
    three resolved that through `LocalArtifactStore.path`, which proves the path sits under a local
    root that is required to be OUTSIDE the worktree — while `runs/` is inside it by design. So all
    three were unreachable for every run this project has ever produced, and it did not look like a
    defect: `attribute` had simply never produced a proposal.

    The single-file fix shipped for `acr audit run` and was wired nowhere else. Reachability is not
    a property one call site can hold.

    THE CHECK THAT SURVIVES is the one that was always the point: Git must not TRACK the record. It
    is applied to EVERY member of a directory, not to the directory itself — a batch is scored as a
    whole, so one tracked member is a disclosure inside the thing being read.

    An EMPTY directory refuses. Returning it would let a mistyped path score zero runs and report a
    clean batch, which is the failure mode this codebase already names as inert-versus-satisfied.
    """
    raw = Path(value).expanduser()
    resolved = (raw if raw.is_absolute() else Path.cwd() / raw).resolve(strict=False)
    if resolved.is_file():
        return require_run_artifact(resolved, what=what)
    if not resolved.is_dir():
        raise LocalArtifactError(f"{what} not found: {resolved}")
    members = sorted(resolved.rglob(RUN_RECORD_GLOB))
    if not members:
        raise LocalArtifactError(
            f"no run record under {resolved}: nothing matches {RUN_RECORD_GLOB}. An empty batch "
            f"scores zero runs and reads like a clean one, so it refuses instead.")
    for m in members:
        require_run_artifact(m, what=f"{what} member")
    return resolved


def _is_git_tracked(path: Path) -> bool:
    """True when Git tracks this exact path. False when Git is unavailable or the path is outside."""
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                           cwd=path.parent, capture_output=True, text=True, check=False)
    except OSError:
        return False
    return r.returncode == 0


class LocalArtifactStore:
    """A mode-0700 directory outside Git, with mode-0600 atomic files."""

    def __init__(self, root: str | Path | None = None):
        supplied = str(root or os.environ.get(LOCAL_ROOT_ENV, "")).strip()
        if not supplied:
            raise LocalArtifactError(
                f"--local-root or {LOCAL_ROOT_ENV} is required for patient-derived artifacts"
            )
        raw = Path(supplied).expanduser()
        if not raw.is_absolute():
            raise LocalArtifactError(f"local artifact root must be absolute: {raw}")
        resolved = raw.resolve(strict=False)
        git = _git_root()
        if resolved == git or _within(resolved, git):
            raise LocalArtifactError(
                f"local artifact root resolves inside the Git worktree: {resolved}"
            )
        self.root = resolved
        self.git_root = git

    def ensure(self) -> Path:
        """Create the already-validated root and enforce private directory permissions."""
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.is_dir():
            raise LocalArtifactError(f"local artifact root is not a directory: {self.root}")
        os.chmod(self.root, 0o700)
        return self.root

    def path(self, value: str | Path, *, must_exist: bool = False,
             what: str = "artifact") -> Path:
        """Resolve an absolute or root-relative path and prove it remains in this store."""
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        if not _within(resolved, self.root):
            raise LocalArtifactError(
                f"{what} must resolve under local root {self.root}: {resolved}"
            )
        if resolved == self.root:
            raise LocalArtifactError(f"{what} must name a file below local root, not the root")
        if must_exist and not resolved.is_file():
            raise LocalArtifactError(f"{what} not found in local root: {resolved}")
        return resolved

    def require_input(self, value: str | Path, *, what: str) -> Path:
        """An existing DEVELOP artifact, which must live under the local root.

        Right for gold, answer keys, registry references, case maps, proposals and sealed sets:
        every one is patient-derived material a human curated, and the point of the store is that
        such a file never sits where Git can reach it.

        WRONG FOR A RUN RECORD, and that mistake made a whole plane unreachable — see
        `require_run_artifact`.
        """
        return self.path(value, must_exist=True, what=what)

    def directory(self, value: str | Path) -> Path:
        path = self.path(value, what="artifact directory")
        self.ensure()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        return path

    def write_json(self, value: str | Path, document: Any) -> Path:
        path = self.path(value, what="output")
        self.ensure()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        payload = json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n"
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            os.chmod(path, 0o600)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        return path

    def append_jsonl(self, value: str | Path, event: Any, *, idempotency_key: str) -> bool:
        """Append one event unless its stable key already exists.

        The files are deliberately small DEVELOP-plane ledgers.  Scanning before append keeps
        the format plain JSONL and makes a retried command idempotent without introducing a
        hidden database.
        """
        path = self.path(value, what="JSONL ledger")
        self.ensure()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("event_id") or "") == idempotency_key:
                    return False
        row = dict(event)
        row["event_id"] = idempotency_key
        payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, 0o600)
        return True


def content_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

