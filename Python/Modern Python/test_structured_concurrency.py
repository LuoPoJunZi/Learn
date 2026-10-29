"""Standard-library tests for the offline structured concurrency example."""

from __future__ import annotations

import asyncio
import unittest

from structured_concurrency import Job, run_batch


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_preserves_input_order(self) -> None:
        events: list[str] = []
        result = await run_batch([Job("slow", 0.02), Job("fast", 0)], 2.0, events)
        self.assertEqual(result, ["SLOW", "FAST"])
        self.assertLess(events.index("done:fast"), events.index("done:slow"))
        self.assertEqual(sum(item.startswith("cleanup:") for item in events), 2)

    async def test_failure_cancels_sibling_and_cleans_up(self) -> None:
        events: list[str] = []
        with self.assertRaises(ExceptionGroup) as caught:
            await run_batch([Job("broken", 0, fail=True), Job("waiting", 60)], 2.0, events)
        self.assertTrue(all(isinstance(e, ValueError) for e in caught.exception.exceptions))
        self.assertIn("cleanup:broken", events)
        self.assertIn("cleanup:waiting", events)
        self.assertNotIn("done:waiting", events)

    async def test_timeout_cleans_up(self) -> None:
        events: list[str] = []
        with self.assertRaises(TimeoutError):
            await run_batch([Job("waiting", 60)], 0.02, events)
        self.assertEqual(events, ["start:waiting", "cleanup:waiting"])

    async def test_external_cancellation_is_not_swallowed(self) -> None:
        events: list[str] = []
        task = asyncio.create_task(run_batch([Job("waiting", 60)], 120, events))
        while "start:waiting" not in events:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn("cleanup:waiting", events)

    async def test_empty_batch(self) -> None:
        self.assertEqual(await run_batch([], 1.0, []), [])

    async def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            await run_batch([], 0, [])
        with self.assertRaises(ValueError):
            await run_batch([Job("invalid", -1)], 1.0, [])


if __name__ == "__main__":
    unittest.main()
