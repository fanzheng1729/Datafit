"""Version-aware local dependency path setup for Datafit scripts.

The repository is meant to run on machines that may already have NumPy/SciPy,
but the development workspace also keeps optional local wheel installs.  This
module finds a compatible local install without making those wheels part of the
GitHub repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_local_deps(anchor: str | Path) -> Path | None:
    """Prepend a compatible local dependency directory, if one exists.

    Binary packages such as NumPy and SciPy are Python-ABI specific.  This
    helper avoids loading a stale ``.python_deps`` directory that was built for
    a different interpreter, e.g. CPython 3.12 wheels under Python 3.14.
    """

    root = Path(anchor).resolve().with_name(".python_deps")
    candidates = _candidate_paths(root)
    for path in candidates:
        if path.is_dir() and _looks_compatible(path):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
            return path
    return None


def current_cp_tag() -> str:
    """Return the CPython ABI tag used in wheel/platform directory names."""

    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _candidate_paths(root: Path) -> list[Path]:
    """Return local dependency paths from most specific to most general."""

    paths: list[Path] = []
    env_path = os.environ.get("DATAFIT_PYTHON_DEPS")
    if env_path:
        # Let callers override everything, which is useful for one-off testing
        # without editing the repo or environment-specific scripts.
        paths.append(Path(env_path))

    cp_tag = current_cp_tag()
    roots = [root.with_name(".python_deps_fresh"), root]
    for dependency_root in roots:
        paths.extend(
            [
                dependency_root / cp_tag,
                dependency_root / f"py{sys.version_info.major}{sys.version_info.minor}",
                dependency_root / (sys.implementation.cache_tag or cp_tag),
                dependency_root,
            ]
        )

    # Preserve priority order while avoiding repeated resolved paths.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _looks_compatible(path: Path) -> bool:
    """Return whether a dependency path is safe for this Python interpreter."""

    if not _has_installed_package_markers(path):
        return False

    cp_tag = current_cp_tag()
    cache_tag = sys.implementation.cache_tag or ""
    found_extensions = False
    try:
        # Compiled extensions encode their CPython ABI tag in the filename.  If
        # extensions exist but none match the current interpreter, importing the
        # directory would fail later with a much less helpful DLL/ABI error.
        for extension in path.rglob("*.pyd"):
            found_extensions = True
            name = extension.name
            if cp_tag in name or cache_tag in name:
                return True
    except OSError:
        return False
    return not found_extensions


def _has_installed_package_markers(path: Path) -> bool:
    """Reject interrupted target installs with dist-info but no import package."""

    try:
        for package in ("numpy", "scipy"):
            has_dist_info = any(path.glob(f"{package}-*.dist-info"))
            if has_dist_info and not (path / package / "__init__.py").is_file():
                return False
    except OSError:
        return False
    return True
