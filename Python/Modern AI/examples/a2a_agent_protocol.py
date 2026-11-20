"""Demonstrate Agent-to-Agent (A2A) Agent Card discovery and task lifecycle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

TERMINAL_STATES = frozenset({"completed", "failed", "canceled"})
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "submitted": frozenset({"working", "canceled", "failed"}),
    "working": frozenset({"completed", "failed", "canceled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "canceled": frozenset(),
}

_AGENT_CARD: dict[str, Any] = {
    "name": "learn-summary-agent",
    "description": "Offline teaching agent that summarizes short notes and counts words.",
    "version": "1.0.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "word-count",
            "name": "Word Counter",
            "description": "Count whitespace-separated tokens in a plain-text note.",
            "tags": ["text", "metrics"],
        },
        {
            "id": "extract-title",
            "name": "Title Extractor",
            "description": "Extract the first non-empty line as a title.",
            "tags": ["text", "summary"],
        },
    ],
}


@dataclass
class TaskRecord:
    """In-memory representation of an A2A delegated task."""

    task_id: str
    skill_id: str
    state: str = "submitted"
    history: list[str] = field(default_factory=lambda: ["submitted"])
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None

    def transition(self, next_state: str) -> None:
        """Advance task state only along legal A2A lifecycle edges."""
        if next_state not in ALLOWED_TRANSITIONS:
            raise ValueError(f"Unknown task state: {next_state!r}")
        if next_state not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"Illegal state transition from {self.state!r} to {next_state!r}."
            )
        self.state = next_state
        self.history.append(next_state)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.task_id,
            "skillId": self.skill_id,
            "status": {"state": self.state},
            "stateHistory": list(self.history),
            "artifacts": copy.deepcopy(self.artifacts),
        }
        if self.error_message is not None:
            payload["status"]["message"] = self.error_message
        return payload


class OfflineA2AAgent:
    """Minimal in-memory A2A agent simulator for learning protocol boundaries."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._skills = {skill["id"] for skill in _AGENT_CARD["skills"]}

    def get_agent_card(self) -> dict[str, Any]:
        """Return a defensive copy of the agent's capability card."""
        return copy.deepcopy(_AGENT_CARD)

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a JSON-RPC 2.0 style A2A request."""
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return self._error(None, -32600, "Invalid Request: expected jsonrpc 2.0 object.")

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._error(req_id, -32602, "Invalid params: expected JSON object.")

        if method == "agent/card":
            return {"jsonrpc": "2.0", "id": req_id, "result": self.get_agent_card()}
        if method == "tasks/send":
            return self._handle_send(req_id, params)
        if method == "tasks/get":
            return self._handle_get(req_id, params)
        if method == "tasks/cancel":
            return self._handle_cancel(req_id, params)
        return self._error(req_id, -32601, f"Method not found: {method!r}")

    def _handle_send(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        skill_id = params.get("skillId")
        text = params.get("text")
        defer = bool(params.get("defer", False))

        if not isinstance(task_id, str) or not task_id.strip():
            return self._error(req_id, -32602, "Invalid params: 'id' must be a non-empty string.")
        if task_id in self._tasks:
            return self._error(req_id, -32602, f"Duplicate task id: {task_id!r}")
        if not isinstance(skill_id, str) or skill_id not in self._skills:
            return self._error(req_id, -32602, f"Unknown skillId: {skill_id!r}")
        if not isinstance(text, str):
            return self._error(req_id, -32602, "Invalid params: 'text' must be a string.")

        record = TaskRecord(task_id=task_id, skill_id=skill_id)
        self._tasks[task_id] = record

        if defer:
            # Leave the task in 'working' state so callers can practice tasks/cancel.
            record.transition("working")
            return {"jsonrpc": "2.0", "id": req_id, "result": record.to_dict()}

        record.transition("working")
        if not text.strip():
            record.error_message = "Input text must contain at least one non-whitespace character."
            record.transition("failed")
            return {"jsonrpc": "2.0", "id": req_id, "result": record.to_dict()}

        if skill_id == "word-count":
            count = len(text.split())
            record.artifacts.append({"name": "metrics", "data": {"wordCount": count}})
        elif skill_id == "extract-title":
            first_line = next(line.strip() for line in text.splitlines() if line.strip())
            record.artifacts.append({"name": "title", "data": {"title": first_line}})

        record.transition("completed")
        return {"jsonrpc": "2.0", "id": req_id, "result": record.to_dict()}

    def _handle_get(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or task_id not in self._tasks:
            return self._error(req_id, -32001, f"Task not found: {task_id!r}")
        return {"jsonrpc": "2.0", "id": req_id, "result": self._tasks[task_id].to_dict()}

    def _handle_cancel(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str) or task_id not in self._tasks:
            return self._error(req_id, -32001, f"Task not found: {task_id!r}")
        record = self._tasks[task_id]
        if record.state in TERMINAL_STATES:
            return self._error(
                req_id,
                -32002,
                f"Task {task_id!r} is already in terminal state {record.state!r}.",
            )
        record.transition("canceled")
        return {"jsonrpc": "2.0", "id": req_id, "result": record.to_dict()}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


def main() -> int:
    agent = OfflineA2AAgent()
    print("Agent-to-Agent (A2A) protocol offline teaching demo\n")

    card = agent.handle_request({"jsonrpc": "2.0", "id": 1, "method": "agent/card"})
    skills = [skill["id"] for skill in card["result"]["skills"]]
    print(f"1. Discovered Agent Card: {card['result']['name']} | skills={skills}")

    completed = agent.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/send",
            "params": {
                "id": "task-101",
                "skillId": "word-count",
                "text": "A2A lets independent agents delegate tasks safely",
            },
        }
    )
    print("2. Delegated task completed:", completed["result"])

    failed = agent.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tasks/send",
            "params": {"id": "task-102", "skillId": "extract-title", "text": "   "},
        }
    )
    print("3. Empty input led to task failure:", failed["result"]["status"])

    deferred = agent.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tasks/send",
            "params": {
                "id": "task-103",
                "skillId": "word-count",
                "text": "long running task",
                "defer": True,
            },
        }
    )
    canceled = agent.handle_request(
        {"jsonrpc": "2.0", "id": 5, "method": "tasks/cancel", "params": {"id": "task-103"}}
    )
    print(
        f"4. Deferred task {deferred['result']['id']} canceled:",
        canceled["result"]["stateHistory"],
    )

    rejected = agent.handle_request(
        {"jsonrpc": "2.0", "id": 6, "method": "tasks/cancel", "params": {"id": "task-101"}}
    )
    print("5. Canceling already completed task rejected:", rejected["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
