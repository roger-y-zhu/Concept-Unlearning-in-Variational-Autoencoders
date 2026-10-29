"""
src/unlearning/provenance.py
=============================
Record what produced a run, beside the run.

WHY
---
Before this, the only record of an experiment's configuration was its directory
name (`a200.0_c1.0_ep20_s5`). That is lossy: it omits every flag left at its
argparse default -- `--lr`, `--val-tolerance`, `--fit-samples`, `--max-batches`
-- and it says nothing about which code, which seeds, or which environment
produced the weights. Changing a default therefore retroactively changes the
meaning of every existing directory name, silently.

The cost of that was real. Establishing whether the dSprites RMU champion was
stale required re-running it and comparing loss curves epoch by epoch, because
nothing in the artifact recorded the code version. A stored commit SHA would
have answered it immediately.

WHAT IS RECORDED
----------------
The fully resolved argument namespace (not just what was typed), all three
seeds, the git commit and whether the tree was dirty, the environment, and the
timestamp. Written once at run start to `run_meta.json`, so it exists even if
the run is later interrupted.

`git_dirty` matters as much as `git_commit`: a SHA from a dirty tree does not
identify the code that ran, and pretending otherwise is worse than recording
nothing. Read it before trusting a commit hash to explain a result.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str) -> str | None:
    """Run a git command in the repo, or return None if git/repo is unavailable."""
    try:
        root = Path(__file__).resolve().parent.parent.parent
        out = subprocess.run(("git", *args), cwd=root, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state() -> dict:
    """Commit, branch, and whether the working tree had uncommitted changes."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # None (git unavailable) is distinct from "" (clean) -- do not collapse
        # them, or an unknown state reads as a clean one.
        "dirty": None if status is None else bool(status),
        "dirty_files": None if status is None else len(status.splitlines()),
    }


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        env.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
        })
    except Exception:
        pass
    for mod in ("numpy", "scipy", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def _jsonable(value):
    """argparse namespaces hold Paths and other non-JSON types."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def write_run_meta(out_dir: Path, *, args, method_name: str, dataset: str,
                   run_name: str, seeds: dict, extra: dict | None = None) -> Path:
    """Write run_meta.json into out_dir and return its path.

    Called at run START, before training, so an interrupted run still says what
    it was trying to do. Overwrites on re-run, which is correct: the file
    describes the attempt that produced the weights currently in the directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": run_name,
        "dataset": dataset,
        "method": method_name,
        # The RESOLVED namespace, so defaults are captured explicitly and a
        # later change to a default cannot rewrite the past.
        "args": {k: _jsonable(v) for k, v in sorted(vars(args).items())},
        "seeds": seeds,
        "git": git_state(),
        "environment": environment(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_schema": 1,
    }
    if extra:
        meta.update({k: _jsonable(v) for k, v in extra.items()})
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def read_run_meta(run_dir: Path) -> dict | None:
    """run_meta.json for a run, or None for runs predating this module."""
    p = Path(run_dir) / "run_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
