from __future__ import annotations

import ast
from pathlib import Path


def test_causalis_has_no_imports_from_tests() -> None:
    root = Path(__file__).resolve().parents[1] / "causalis"
    violations: list[str] = []

    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name == "tests" or name.startswith("tests."):
                        violations.append(f"{path}: import {name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "tests" or module.startswith("tests."):
                    violations.append(f"{path}: from {module} import ...")

    assert not violations, "Imports from tests found in causalis:\\n" + "\\n".join(violations)
