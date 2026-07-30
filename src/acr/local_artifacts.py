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

