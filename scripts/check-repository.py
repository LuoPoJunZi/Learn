"""Check repository structure and portable path conventions."""

from __future__ import annotations

import argparse
import subprocess
import unicodedata
from pathlib import Path, PurePosixPath


REQUIRED_ROOT_FILES = {
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
}

FORBIDDEN_DIRECTORIES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "result",
}

FORBIDDEN_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".asv",
    ".bak",
    ".log",
    ".pyc",
    ".pyo",
    ".rar",
    ".tmp",
    ".zip",
}

WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def repository_paths(root: Path) -> list[PurePosixPath]:
    """Return tracked and untracked non-ignored paths in NUL format."""

    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {message}")

    decoded = result.stdout.decode("utf-8")
    candidates = {
        PurePosixPath(item) for item in decoded.split("\0") if item
    }
    existing = [
        path for path in candidates if root.joinpath(*path.parts).exists()
    ]
    return sorted(existing, key=lambda path: path.as_posix())


def portable_key(path: PurePosixPath) -> str:
    """Normalize a path for case-insensitive cross-platform comparison."""

    normalized = unicodedata.normalize("NFC", path.as_posix())
    return normalized.casefold()


def audit_paths(paths: list[PurePosixPath]) -> list[str]:
    """Return repository layout issues for tracked paths."""

    issues: list[str] = []
    path_set = {path.as_posix() for path in paths}

    for required in sorted(REQUIRED_ROOT_FILES - path_set):
        issues.append(f"missing required root file: {required}")

    if "AGENTS.md" in path_set:
        issues.append("AGENTS.md is local-only and must not be tracked")

    top_directories = {
        path.parts[0]
        for path in paths
        if len(path.parts) > 1 and not path.parts[0].startswith(".")
    }
    for directory in sorted(top_directories):
        readme = f"{directory}/README.md"
        if readme not in path_set:
            issues.append(f"top-level directory is missing README.md: {directory}")

    normalized_paths: dict[str, PurePosixPath] = {}
    for path in paths:
        path_text = path.as_posix()
        key = portable_key(path)
        previous = normalized_paths.get(key)
        if previous is not None and previous != path:
            issues.append(
                "portable path collision: "
                f"{previous.as_posix()} <-> {path_text}"
            )
        else:
            normalized_paths[key] = path

        for part in path.parts:
            if part.endswith((" ", ".")):
                issues.append(f"path component ends with a space or dot: {path_text}")
            if any(ord(character) < 32 for character in part):
                issues.append(f"path contains a control character: {path_text}")

            base_name = part.rstrip(" .").split(".", maxsplit=1)[0].casefold()
            if base_name in WINDOWS_RESERVED_NAMES:
                issues.append(f"path uses a Windows reserved name: {path_text}")

        if FORBIDDEN_DIRECTORIES.intersection(path.parts):
            issues.append(f"generated directory is tracked: {path_text}")
        if path.name in FORBIDDEN_FILE_NAMES:
            issues.append(f"generated file is tracked: {path_text}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            issues.append(
                f"temporary, generated, or source archive file is tracked: {path_text}"
            )
        if path.suffix and path.suffix != path.suffix.casefold():
            issues.append(f"file extension must be lowercase: {path_text}")

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

    try:
        paths = repository_paths(root)
    except (OSError, RuntimeError, UnicodeDecodeError) as exc:
        print(f"Repository checks failed: {exc}")
        return 1

    issues = audit_paths(paths)
    if issues:
        print("Repository checks failed:")
        for issue in issues:
            print(f"  {issue}")
        return 1

    print(f"Repository checks passed: {len(paths)} versioned or pending files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
