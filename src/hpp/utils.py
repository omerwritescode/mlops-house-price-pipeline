"""Small shared helpers: hashing (data versions), JSON I/O, git info, logging, atomic directory swaps."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


# ------------------------------------------------------------------ hashing
def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def combined_hash(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def dvc_md5(path: str | Path) -> str | None:
    """Return the md5 recorded by DVC for ``path`` if it is DVC-tracked (``<path>.dvc`` exists)."""
    dvc_file = Path(str(path) + ".dvc")
    if not dvc_file.exists():
        return None
    try:
        import yaml

        meta = yaml.safe_load(dvc_file.read_text()) or {}
        outs = meta.get("outs") or [{}]
        return outs[0].get("md5")
    except Exception:  # pragma: no cover - best effort
        return None


# ------------------------------------------------------------------ JSON
class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, cls=_Encoder, sort_keys=False))
    os.replace(tmp, p)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(obj, cls=_Encoder) + "\n")


# ------------------------------------------------------------------ git
def _git(*args: str, root: Path | None = None) -> str | None:
    cmd = ["git", "-C", str(root), *args] if root else ["git", *args]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_info(root: Path | None = None) -> dict[str, Any]:
    """Code version for lineage (of the repository at ``root``). CI variables win over the local checkout."""
    commit = (os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA") or _git("rev-parse", "HEAD", root=root)
              or "unknown")
    branch = (os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
              or _git("rev-parse", "--abbrev-ref", "HEAD", root=root) or "unknown")
    status = _git("status", "--porcelain", "--untracked-files=no", root=root)
    return {
        "git_commit": commit,
        "git_short": commit[:7] if commit != "unknown" else "unknown",
        "git_branch": branch,
        "git_dirty": bool(status) if status is not None else None,
    }


# ------------------------------------------------------------------ logging
def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_hpp", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._hpp = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("matplotlib", "urllib3", "alembic", "git", "httpx", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class StageLogFile:
    """Context manager that tees all ``hpp`` log records into ``logs/<run_id>/<stage>.log``."""

    def __init__(self, path: Path):
        self.path = path
        self.handler: logging.Handler | None = None

    def __enter__(self) -> StageLogFile:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(self.path)
        self.handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger("hpp").addHandler(self.handler)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handler:
            logging.getLogger("hpp").removeHandler(self.handler)
            self.handler.close()


# ------------------------------------------------------------------ filesystem
def replace_dir_atomically(staging: Path, current: Path, previous: Path | None) -> None:
    """Swap ``staging`` into ``current``; keep the old ``current`` as ``previous`` (if given)."""
    if previous is not None:
        if previous.exists():
            shutil.rmtree(previous)
        if current.exists():
            os.replace(current, previous)
    elif current.exists():
        shutil.rmtree(current)
    os.replace(staging, current)


def dir_fingerprint(path: Path) -> str | None:
    """Hash of every file in a directory - used to prove a deployment did *not* change."""
    if not path.exists():
        return None
    parts = [f"{p.relative_to(path)}:{file_sha256(p)}" for p in sorted(path.rglob("*")) if p.is_file()]
    return combined_hash(parts)[:16]
