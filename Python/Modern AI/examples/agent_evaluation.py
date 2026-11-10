"""Offline evaluation of synthetic agent runs; no model, telemetry upload, or secrets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class ToolEvent:
    """Record outcomes, not raw arguments or private tool responses."""

    tool: str
    ok: bool


@dataclass(frozen=True)
class Case:
    case_id: str
    expected: str
    required_tools: tuple[str, ...] = ()
    max_calls: int = 2


@dataclass(frozen=True)
class Run:
    case_id: str
    answer: str
    events: tuple[ToolEvent, ...]
    latency_ms: float
    error: str | None = None


def evaluate(cases: list[Case], runs: list[Run]) -> dict[str, float | int]:
    """Score exactly one run per case; missing or duplicate runs are invalid input."""
    if not cases:
        raise ValueError("evaluation requires at least one case")
    # Dataclass annotations do not validate values loaded from JSON or other callers.
    for case in cases:
        if not isinstance(case.case_id, str) or not case.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(case.expected, str):
            raise ValueError("expected answers must be strings")
        if not isinstance(case.required_tools, (tuple, list)) or any(
            not isinstance(tool, str) or not tool.strip() for tool in case.required_tools
        ):
            raise ValueError("required_tools must be a sequence of non-empty tool names")
    for run in runs:
        if not isinstance(run.case_id, str) or not run.case_id.strip():
            raise ValueError("run case_id must be a non-empty string")
        if not isinstance(run.answer, str):
            raise ValueError("run answers must be strings")
        if run.error is not None and (not isinstance(run.error, str) or not run.error.strip()):
            raise ValueError("error must be None or a non-empty error category")
        if not isinstance(run.events, (tuple, list)) or any(
            not isinstance(event, ToolEvent) or type(event.ok) is not bool
            or not isinstance(event.tool, str) or not event.tool.strip()
            for event in run.events
        ):
            raise ValueError("events require non-empty tool names and boolean outcomes")
    case_map = {case.case_id: case for case in cases}
    run_map = {run.case_id: run for run in runs}
    if len(case_map) != len(cases) or len(run_map) != len(runs):
        raise ValueError("case IDs and run IDs must be unique")
    if set(case_map) != set(run_map):
        raise ValueError("runs must match the evaluation cases exactly")
    if any(type(case.max_calls) is not int or case.max_calls < 0 for case in cases):
        raise ValueError("max_calls must be a non-negative integer")
    if any(type(run.latency_ms) not in (int, float)
           or not 0 <= run.latency_ms <= 1e15
           or not math.isfinite(run.latency_ms) for run in runs):
        raise ValueError("latencies must be numeric, finite, and in [0, 1e15] milliseconds")

    exact_matches = 0
    task_successes = 0
    tool_calls = 0
    failed_tool_calls = 0
    for case_id, case in case_map.items():
        run = run_map[case_id]
        # Only trim surrounding whitespace; do not silently change units or case.
        correct = run.error is None and run.answer.strip() == case.expected.strip()
        successful_tools = {event.tool for event in run.events if event.ok}
        all_tools_ok = all(event.ok for event in run.events)
        required_used = set(case.required_tools) <= successful_tools
        within_budget = len(run.events) <= case.max_calls
        exact_matches += correct
        task_successes += correct and all_tools_ok and required_used and within_budget
        tool_calls += len(run.events)
        failed_tool_calls += sum(not event.ok for event in run.events)

    return {
        "cases": len(cases),
        "answer_exact_match": exact_matches / len(cases),
        "task_success_rate": task_successes / len(cases),
        "tool_calls": tool_calls,
        "failed_tool_calls": failed_tool_calls,
        # None of the calls failed when no calls exist; counts keep that visible.
        "tool_error_rate": failed_tool_calls / tool_calls if tool_calls else 0.0,
        "mean_latency_ms": mean(run.latency_ms for run in runs),
    }


def fixtures() -> tuple[list[Case], list[Run], list[Run]]:
    """Use the same frozen cases for both versions; latency numbers are synthetic."""
    cases = [
        Case("sum", "4", ("calculator",)),
        Case("lookup", "guide-uv", ("search",)),
        Case("refuse", "REFUSE", max_calls=0),
    ]
    baseline = [
        Run("sum", "4", (ToolEvent("calculator", True),), 10),
        Run("lookup", "guide-uv", (ToolEvent("search", False),), 30),
        Run("refuse", "REFUSE", (), 5),
    ]
    candidate = [
        Run("sum", "4", (ToolEvent("calculator", True),), 12),
        Run("lookup", "guide-uv", (ToolEvent("search", True),), 35),
        Run("refuse", "REFUSE", (), 6),
    ]
    return cases, baseline, candidate


def main() -> int:
    import json

    cases, baseline, candidate = fixtures()
    for name, runs in [("baseline", baseline), ("candidate", candidate)]:
        print(name, json.dumps(evaluate(cases, runs), indent=2, allow_nan=False))
    print("Synthetic fixtures are not evidence of real agent quality or speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
