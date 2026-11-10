"""Standard-library checks for migration semantics, agent evaluation, and optional DuckDB."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def load_example(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluation = load_example("agent_evaluation", "Python/Modern AI/examples/agent_evaluation.py")
migration = load_example("migration_checks", "Python/Modern Python/migration_checks.py")
sales = load_example("duckdb_sales", "Python/Data Analysis/duckdb_sales.py")


class EvaluationTests(unittest.TestCase):
    def test_fixture_metrics_separate_answer_and_task_success(self) -> None:
        cases, baseline, candidate = evaluation.fixtures()
        first = evaluation.evaluate(cases, baseline)
        second = evaluation.evaluate(cases, candidate)
        self.assertEqual(first["answer_exact_match"], 1.0)
        self.assertAlmostEqual(first["task_success_rate"], 2 / 3)
        self.assertEqual(second["task_success_rate"], 1.0)
        self.assertEqual(first["failed_tool_calls"], 1)
        self.assertEqual(first["tool_error_rate"], 0.5)

    def test_missing_required_tool_and_budget(self) -> None:
        case = evaluation.Case("a", "4", ("calculator",), max_calls=1)
        for events in [(), (evaluation.ToolEvent("calculator", True),) * 2]:
            run = evaluation.Run("a", "4", events, 0)
            self.assertEqual(evaluation.evaluate([case], [run])["task_success_rate"], 0)

    def test_runtime_error_and_wrong_answer(self) -> None:
        case = evaluation.Case("a", "4")
        for run in [evaluation.Run("a", "4", (), 1, "timeout"),
                    evaluation.Run("a", "5", (), 1)]:
            self.assertEqual(evaluation.evaluate([case], [run])["answer_exact_match"], 0)

    def test_missing_extra_and_duplicate_runs_are_not_silent(self) -> None:
        case = evaluation.Case("a", "4")
        run = evaluation.Run("a", "4", (), 1)
        for cases, runs in [([], []), ([case], []), ([case], [run, run]),
                            ([case, case], [run]),
                            ([case], [evaluation.Run("b", "4", (), 1)])]:
            with self.assertRaises(ValueError):
                evaluation.evaluate(cases, runs)

    def test_invalid_latency_and_budget(self) -> None:
        for latency in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                evaluation.evaluate([evaluation.Case("a", "4")],
                                    [evaluation.Run("a", "4", (), latency)])
        with self.assertRaises(ValueError):
            evaluation.evaluate([evaluation.Case("a", "4", max_calls=-1)],
                                [evaluation.Run("a", "4", (), 1)])

    def test_no_tool_calls_and_whitespace(self) -> None:
        result = evaluation.evaluate([evaluation.Case("a", "4")],
                                     [evaluation.Run("a", " 4\n", (), 0)])
        self.assertEqual(result["task_success_rate"], 1.0)
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["tool_error_rate"], 0)


class MigrationTests(unittest.TestCase):
    def test_omitted_none_and_falsy_values_are_distinct(self) -> None:
        self.assertEqual(migration.describe_update(), "keep current value")
        self.assertEqual(migration.describe_update(None), "clear current value")
        self.assertEqual(migration.describe_update(0), "set value: 0")
        self.assertEqual(migration.describe_update(""), "set value: ")


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "optional DuckDB is not installed")
class DuckDBTests(unittest.TestCase):
    def setUp(self) -> None:
        import duckdb

        self.connection = duckdb.connect(":memory:")
        self.addCleanup(self.connection.close)
        sales.load_sales(self.connection)

    def test_summary_and_filter(self) -> None:
        self.assertEqual(sales.summary(self.connection), [
            ("courses", 3, 19700), ("books", 3, 11500), ("tools", 3, 3600),
        ])
        self.assertEqual(sales.summary(self.connection, 4), [])
        self.assertEqual(sales.summary(self.connection, 2)[0], ("courses", 2, 9800))

    def test_parquet_roundtrip_in_temporary_directory(self) -> None:
        size, restored = sales.parquet_roundtrip(self.connection)
        self.assertGreater(size, 0)
        self.assertEqual(restored, sales.summary(self.connection))

    def test_invalid_threshold(self) -> None:
        for threshold in [0, -1, True, "1 OR 1=1"]:
            with self.assertRaises(ValueError):
                sales.summary(self.connection, threshold)


if __name__ == "__main__":
    unittest.main()
