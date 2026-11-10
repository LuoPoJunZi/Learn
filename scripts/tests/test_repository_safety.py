"""Regression tests for checkers and file tools, restricted to temporary fixtures."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repository = load("audit_repository", "scripts/check-repository.py")
html = load("audit_html", "scripts/check-html-code.py")
javascript = load("audit_javascript", "scripts/check-javascript.py")
evaluation = load("audit_evaluation", "Python/Modern AI/examples/agent_evaluation.py")
rename = load("audit_rename", "Python/Auto_scripts/rename_files/rename_files.py")
sort = load("audit_sort", "Python/Auto_scripts/sort_files/sort_files.py")
empty = load("audit_empty", "Python/Auto_scripts/remove_empty_folders/remove_empty_folders.py")


class CheckerTests(unittest.TestCase):
    def test_directory_case_collision(self) -> None:
        issues = repository.audit_paths([PurePosixPath("Foo/a.py"), PurePosixPath("foo/b.py")])
        self.assertTrue(any("portable path collision" in issue for issue in issues))

    def test_numbered_environment_is_forbidden(self) -> None:
        issues = repository.audit_paths([PurePosixPath(".venv315/pyvenv.cfg")])
        self.assertTrue(any("generated directory" in issue for issue in issues))

    def test_inline_scripts_keep_raw_source_and_skip_data(self) -> None:
        parser = javascript.ScriptCollector()
        parser.feed('<script>const x = "&amp;";</script>'
                    '<script type="application/json">{"x": 1}</script>'
                    '<script src="external.js"></script>'
                    '<script type="module">await Promise.resolve();</script>')
        self.assertEqual(len(parser.blocks), 2)
        self.assertIn('"&amp;"', parser.blocks[0][2])
        self.assertEqual(parser.blocks[1][1], "module")

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_javascript_check_does_not_execute(self) -> None:
        node = shutil.which("node")
        self.assertIsNone(javascript.check_source(node, 'throw new Error("not executed");',
                                                 "commonjs"))
        self.assertIsNotNone(javascript.check_source(node, "const broken = ;", "commonjs"))
        self.assertIsNone(javascript.check_source(node, "await Promise.resolve();", "module"))

    def test_missing_local_assets_and_url_queries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.js").write_text("", encoding="utf-8")
            page = root / "index.html"
            page.write_text('<script src="valid.js?v=1"></script>'
                            '<link rel="stylesheet" href="missing.css">'
                            '<img src="https://example.org/image.png" alt="example">',
                            encoding="utf-8")
            issues = html.audit_file(page)
            self.assertEqual(len(issues), 1)
            self.assertIn("missing.css", issues[0])


class EvaluationInputTests(unittest.TestCase):
    def test_reject_truthy_false_string(self) -> None:
        case = evaluation.Case("a", "4", ("calculator",))
        run = evaluation.Run("a", "4", (evaluation.ToolEvent("calculator", "false"),), 1)
        with self.assertRaises(ValueError):
            evaluation.evaluate([case], [run])

    def test_reject_invalid_fields(self) -> None:
        case = evaluation.Case("a", "4")
        for run in [evaluation.Run("a", 4, (), 1), evaluation.Run("a", "4", (), True),
                    evaluation.Run("a", "4", (), "1"), evaluation.Run("a", "4", (), 1, "")]:
            with self.assertRaises(ValueError):
                evaluation.evaluate([case], [run])
        with self.assertRaises(ValueError):
            evaluation.evaluate([evaluation.Case("a", "4", "calculator")],
                                [evaluation.Run("a", "4", (), 1)])


class FileSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory(prefix="learn-safety-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_rename_rejects_path_escape(self) -> None:
        source = self.root / "old.txt"
        source.write_text("keep", encoding="utf-8")
        for replacement in ["../escape", "..\\escape"]:
            with self.assertRaises(ValueError):
                rename.rename_files(str(self.root), "old", replacement, dry_run=False)
        self.assertEqual(source.read_text(encoding="utf-8"), "keep")

    def test_rename_preview_and_collision(self) -> None:
        (self.root / "old.txt").write_text("original", encoding="utf-8")
        rename.rename_files(str(self.root), "old", "new")
        self.assertFalse((self.root / "new.txt").exists())
        self.assertFalse((self.root / "rename_files.log").exists())
        (self.root / "new.txt").write_text("existing", encoding="utf-8")
        rename.rename_files(str(self.root), "old", "new", dry_run=False)
        self.assertEqual((self.root / "new.txt").read_text(encoding="utf-8"), "existing")
        self.assertTrue((self.root / "old.txt").exists())

    def test_rename_apply_in_temporary_directory(self) -> None:
        (self.root / "old.txt").write_text("original", encoding="utf-8")
        rename.rename_files(str(self.root), "old", "new", dry_run=False)
        self.assertFalse((self.root / "old.txt").exists())
        self.assertEqual((self.root / "new.txt").read_text(encoding="utf-8"), "original")

    def test_sort_rejects_symlink_destination_without_requiring_os_privileges(self) -> None:
        (self.root / "demo.txt").write_text("keep", encoding="utf-8")
        with patch.object(sort.os.path, "islink", side_effect=lambda p: Path(p).name == "txt"):
            sort.sort_files(str(self.root), dry_run=False)
        self.assertTrue((self.root / "demo.txt").exists())
        self.assertFalse((self.root / "txt").exists())

    def test_sort_preview_and_apply(self) -> None:
        (self.root / "demo.txt").write_text("keep", encoding="utf-8")
        sort.sort_files(str(self.root))
        self.assertFalse((self.root / "txt").exists())
        sort.sort_files(str(self.root), dry_run=False)
        self.assertEqual((self.root / "txt" / "demo.txt").read_text(encoding="utf-8"), "keep")

    def test_empty_parent_appears_in_preview_and_apply(self) -> None:
        parent = self.root / "parent"
        (parent / "child").mkdir(parents=True)
        empty.remove_empty_folders(str(self.root))
        self.assertIn(f"将删除空文件夹 {parent}\n", self.output.getvalue())
        self.assertTrue((parent / "child").exists())
        empty.remove_empty_folders(str(self.root), dry_run=False)
        self.assertFalse(parent.exists())
        self.assertTrue(self.root.exists())

    def test_nonempty_folder_is_preserved(self) -> None:
        parent = self.root / "parent"
        parent.mkdir()
        (parent / "keep.txt").write_text("keep", encoding="utf-8")
        empty.remove_empty_folders(str(self.root), dry_run=False)
        self.assertTrue((parent / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
