"""Offline TaskGroup demo: success, failure cleanup, and whole-batch timeout (3.11+)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    """Describe simulated asynchronous work; no network or file access."""

    name: str
    delay: float
    fail: bool = False


async def simulate_job(job: Job, events: list[str]) -> str:
    """Record lifecycle events and always clean up, including on cancellation."""
    events.append(f"start:{job.name}")
    try:
        await asyncio.sleep(job.delay)
        if job.fail:
            raise ValueError(f"simulated failure: {job.name}")
        events.append(f"done:{job.name}")
        return job.name.upper()
    finally:
        # Real code would release a connection or other resource here.
        events.append(f"cleanup:{job.name}")


async def run_batch(jobs: list[Job], timeout: float, events: list[str]) -> list[str]:
    """Return input-ordered results, or propagate failure/timeout after cleanup."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if any(job.delay < 0 for job in jobs):
        raise ValueError("job delay must not be negative")

    async with asyncio.timeout(timeout):
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(simulate_job(job, events)) for job in jobs]
    return [task.result() for task in tasks]


async def demo() -> None:
    events: list[str] = []
    results = await run_batch([Job("slow", 0.03), Job("fast", 0.01)], 1.0, events)
    print("success:", results)
    print("events:", events)

    events = []
    try:
        await run_batch([Job("broken", 0.01, fail=True), Job("waiting", 1.0)], 2.0, events)
    except* ValueError as errors:
        print("failure:", [str(error) for error in errors.exceptions])
    print("events:", events)

    events = []
    try:
        await run_batch([Job("waiting", 1.0)], 0.02, events)
    except TimeoutError:
        print("timeout: batch cancelled")
    print("events:", events)


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
