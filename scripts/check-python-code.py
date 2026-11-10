"""Run dependency-free quality checks for Python examples in this repository."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


NETWORK_METHODS = {"delete", "get", "head", "options", "patch", "post", "put"}
DEPRECATED_NAMES = {
    "Image." "ANTIALIAS": "use Image.Resampling.LANCZOS",
    "PyPDF2." "PdfFileReader": "use PyPDF2.PdfReader",
    "." "extractText(": "use page.extract_text()",
}


def is_main_guard(node: ast.AST) -> bool:
    """Return whether a node is an ``if __name__ == '__main__'`` guard."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    comparators = node.test.comparators
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(comparators) == 1
        and isinstance(comparators[0], ast.Constant)
        and comparators[0].value == "__main__"
    )


def call_name(node: ast.Call) -> tuple[str | None, str | None]:
    """Return the object and method names for calls such as requests.get()."""
    if not isinstance(node.func, ast.Attribute):
        return None, None
    owner = node.func.value
    if isinstance(owner, ast.Name):
        return owner.id, node.func.attr
    return None, node.func.attr


def text_open_without_encoding(node: ast.Call) -> bool:
    """Detect built-in text-mode open() calls without an explicit encoding."""
    if not isinstance(node.func, ast.Name) or node.func.id != "open":
        return False
    if any(keyword.arg == "encoding" for keyword in node.keywords):
        return False

    mode_node = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
            break

    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return "b" not in mode_node.value
    return mode_node is None


def audit_file(path: Path) -> list[str]:
    """Return human-readable issues found in one Python file."""
    issues: list[str] = []
    try:
        source = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        return [f"cannot decode as UTF-8: {exc}"]

    lines = source.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if "\t" in line:
            issues.append(f"line {line_number}: tab indentation is not allowed")
        if line.rstrip() != line:
            issues.append(f"line {line_number}: trailing whitespace")
        if len(line) > 100:
            issues.append(f"line {line_number}: exceeds 100 characters ({len(line)})")

    if source and not source.endswith("\n"):
        issues.append("file must end with a newline")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [*issues, f"line {exc.lineno}: syntax error: {exc.msg}"]

    has_main_guard = any(is_main_guard(node) for node in tree.body)
    runtime_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.For, ast.Try, ast.While, ast.With))
        or (
            isinstance(node, ast.Expr)
            and not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        )
    ]
    if runtime_nodes and not has_main_guard:
        issues.append("top-level executable code must be protected by a main guard")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                issues.append(f"line {node.lineno}: wildcard imports are not allowed")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(f"line {node.lineno}: bare except is not allowed")
        elif isinstance(node, ast.Call):
            owner, method = call_name(node)
            if owner == "requests" and method in NETWORK_METHODS:
                if not any(keyword.arg == "timeout" for keyword in node.keywords):
                    issues.append(
                        f"line {node.lineno}: requests.{method}() must set timeout"
                    )
            if text_open_without_encoding(node):
                issues.append(f"line {node.lineno}: text open() must set encoding")

    for name, replacement in DEPRECATED_NAMES.items():
        if name in source:
            issues.append(f"deprecated API {name!r}: {replacement}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    ignored_directories = {".agents", ".codex", ".git", "venv", "node_modules"}
    python_files = sorted(
        path
        for path in root.rglob("*.py")
        if ignored_directories.isdisjoint(path.relative_to(root).parts)
        and not any(part.startswith(".venv") for part in path.relative_to(root).parts)
    )

    issue_count = 0
    for path in python_files:
        for issue in audit_file(path):
            issue_count += 1
            print(f"{path.relative_to(root)}:{issue}")

    if issue_count:
        print(f"Python checks failed: {issue_count} issue(s) in {len(python_files)} file(s).")
        return 1

    print(f"Python checks passed: {len(python_files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
