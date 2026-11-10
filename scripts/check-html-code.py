"""Check beginner-facing HTML conventions without third-party dependencies."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class DocumentAudit(HTMLParser):
    """Collect duplicate IDs and missing attributes from one HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: dict[str, list[int]] = {}
        self.images_without_alt: list[int] = []
        self.buttons_without_type: list[int] = []
        self.assets: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        line_number, _ = self.getpos()

        asset = None
        if tag in {"script", "img", "audio", "video", "source"}:
            asset = attributes.get("src")
        if tag == "link" and attributes.get("rel") == "stylesheet":
            asset = attributes.get("href")
        if asset:
            self.assets.append((line_number, asset))

        element_id = attributes.get("id")
        if element_id:
            self.ids.setdefault(element_id, []).append(line_number)
        if tag == "img" and "alt" not in attributes:
            self.images_without_alt.append(line_number)
        if tag == "button" and "type" not in attributes:
            self.buttons_without_type.append(line_number)


def audit_file(path: Path) -> list[str]:
    parser = DocumentAudit()
    try:
        parser.feed(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        return [f"cannot parse HTML: {exc}"]

    issues: list[str] = []
    for element_id, line_numbers in parser.ids.items():
        if len(line_numbers) > 1:
            joined_lines = ", ".join(str(number) for number in line_numbers)
            issues.append(f"duplicate id {element_id!r} on lines {joined_lines}")
    for line_number in parser.images_without_alt:
        issues.append(f"line {line_number}: img must define alt")
    for line_number in parser.buttons_without_type:
        issues.append(f"line {line_number}: button must define type")
    for line_number, asset in parser.assets:
        url = urlsplit(asset)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            continue
        if not (path.parent / unquote(url.path)).is_file():
            issues.append(f"line {line_number}: missing local asset {asset}")
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
    html_files = sorted(
        path
        for path in root.rglob("*.html")
        if ignored_directories.isdisjoint(path.relative_to(root).parts)
        and not any(part.startswith(".venv") for part in path.relative_to(root).parts)
    )

    issue_count = 0
    for path in html_files:
        for issue in audit_file(path):
            issue_count += 1
            print(f"{path.relative_to(root)}:{issue}")

    if issue_count:
        print(f"HTML checks failed: {issue_count} issue(s) in {len(html_files)} file(s).")
        return 1

    print(f"HTML checks passed: {len(html_files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

