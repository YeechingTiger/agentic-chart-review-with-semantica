"""A private on-disk store for patient-derived DEVELOP artifacts.

The store is a mode-0700 directory holding mode-0600 files, written atomically.  Every command
handling registry references, case maps, chart-observable gold, traces, or attribution reports
comes through this module, so that such a file has one resolution path and one permission policy
rather than each caller inventing its own.

The store is deliberately a directory, not a database or dataset registry.  Its contents may
contain PHI even when case identifiers are pseudonymous; nothing here calls them de-identified
or shareable.

WHAT THIS NO LONGER DOES.  Until 2026-08-05 this module also enforced a Git boundary: the store
root was refused if it resolved inside the worktree, and a run record was refused if Git tracked
it.  That boundary was removed deliberately and nothing replaced it, so nothing here prevents
patient-derived material from being committed.  `.gitignore` still covers `runs/` and
`corpus_real/`, and `.gitignore` is not a security boundary: an ignored file can still be
force-added.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

LOCAL_ROOT_ENV = "ACR_LOCAL_ARTIFACT_ROOT"


class LocalArtifactError(ValueError):
    """A sensitive artifact escaped (or could escape) the declared local root."""


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_run_artifact(value: str | Path, *, what: str = "run artifact") -> Path:
    """An existing RUN RECORD — a manifest or a trace — resolved for reading.

    WHY THIS EXISTS SEPARATELY FROM THE STORE. Every command that reads a completed run used
    `LocalArtifactStore.require_input`, which proves a path resolves UNDER the local root. But
    `runs/` is not under the local root — it sits in the worktree, gitignored wholesale — so the
    proof could never succeed and `acr audit run` could not be pointed at any of the 508 manifests
    on disk. The plane looked unexercised; it was unreachable. An adversarial review named this as
    the highest-value fix available, and running the chain end to end reproduced it independently.

    It used to also refuse a record that Git TRACKED, on the grounds that a tracked run record is a
    disclosure that has already happened. That check was removed with the rest of the Git boundary;
    see the module docstring.
    """
    raw = Path(value).expanduser()
    resolved = (raw if raw.is_absolute() else Path.cwd() / raw).resolve(strict=False)
    if not resolved.is_file():
        raise LocalArtifactError(f"{what} not found: {resolved}")
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
        self.root = raw.resolve(strict=False)

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

