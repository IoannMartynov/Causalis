from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_docs_dependencies() -> None:
    pytest.importorskip("sphinx")
    pytest.importorskip("myst_parser")
    pytest.importorskip("autodoc2")


def _copy_repo_subset(tmp_path: Path) -> Path:
    repo_copy = tmp_path / "repo"
    repo_copy.mkdir()

    shutil.copy2(REPO_ROOT / "pyproject.toml", repo_copy / "pyproject.toml")
    shutil.copytree(
        REPO_ROOT / "causalis",
        repo_copy / "causalis",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        REPO_ROOT / "scripts",
        repo_copy / "scripts",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    notebooks_dir = repo_copy / "notebooks"
    notebooks_dir.mkdir()
    shutil.copytree(REPO_ROOT / "notebooks" / "api", notebooks_dir / "api")

    return repo_copy


def _run_generator(repo_copy: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/generate_api_reference.py"],
        cwd=repo_copy,
        capture_output=True,
        text=True,
        check=True,
    )


def _read_html_tree(root: Path) -> dict[str, str]:
    suffixes = {".html", ".js", ".txt"}
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    }


