"""Syntax-check standalone and inline JavaScript without executing application code."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path


class ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[tuple[int, str, str]] = []
        self.current: tuple[int, str] | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "script":
            return
        attributes = dict(attrs)
        script_type = (attributes.get("type") or "").strip().lower()
        supported = {"", "text/javascript", "application/javascript", "module"}
        if "src" not in attributes and script_type in supported:
            self.current = (self.getpos()[0], "module" if script_type == "module" else "commonjs")
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.current is not None:
            line, mode = self.current
            self.blocks.append((line, mode, "".join(self.parts)))
            self.current = None
            self.parts = []


def check_source(node: str, source: str, mode: str) -> str | None:
    result = subprocess.run(
        [node, "--check", f"--input-type={mode}"], input=source,
        text=True, encoding="utf-8", capture_output=True, timeout=30, check=False,
    )
    return result.stderr.strip() if result.returncode else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    node = shutil.which("node")
    if node is None:
        parser.exit(1, "Node.js is required for JavaScript syntax checks.\n")
    root = args.root.resolve()
    failures = 0
    standalone = 0
    inline = 0
    for path in sorted(root.rglob("*")):
        if any(part in {".git", ".agents", ".codex", "venv", "node_modules"}
               or part.startswith(".venv") for part in path.relative_to(root).parts):
            continue
        if path.suffix not in {".js", ".mjs", ".cjs", ".html"} or not path.is_file():
            continue
        source = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".html":
            collector = ScriptCollector()
            collector.feed(source)
            blocks = collector.blocks
            inline += len(blocks)
        else:
            blocks = [(1, "module" if path.suffix == ".mjs" else "commonjs", source)]
            standalone += 1
        for line, mode, content in blocks:
            error = check_source(node, content, mode)
            if error:
                failures += 1
                print(f"{path.relative_to(root)}: script starts at line {line}\n{error}")
    print(f"JavaScript syntax: {standalone} files, {inline} inline blocks, {failures} failures.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
