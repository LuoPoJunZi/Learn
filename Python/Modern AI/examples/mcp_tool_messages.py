"""Offline MCP tools message exercise, NOT a transport or a usable MCP server."""

from __future__ import annotations

import json
import math
from typing import Any


TOOL = {
    "name": "divide",
    "description": "Divide two bounded integers; no network or file access.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "numerator": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
            "denominator": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
        },
        "required": ["numerator", "denominator"],
        "additionalProperties": False,
    },
    "outputSchema": {
        "type": "object",
        "properties": {"quotient": {"type": "number"}},
        "required": ["quotient"],
        "additionalProperties": False,
    },
}


def protocol_error(request_id: str | int | None, code: int, message: str) -> dict[str, Any]:
    """Construct a JSON-RPC error envelope for this fixed request exercise."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def tool_error(message: str) -> dict[str, Any]:
    """Tool execution failure differs from a JSON-RPC protocol error."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def divide(arguments: object) -> dict[str, Any]:
    """Validate only this tool's fixed shape, NOT arbitrary JSON Schema."""
    required = {"numerator", "denominator"}
    if not isinstance(arguments, dict) or set(arguments) != required:
        return tool_error("expected exactly numerator and denominator")
    # JSON Schema integers include 1.0, but exclude booleans and non-finite values.
    if any(
        type(value) not in (int, float)
        or not -1000000 <= value <= 1000000
        or not math.isfinite(value)
        or value != int(value)
        for value in arguments.values()
    ):
        return tool_error("arguments must be integers in [-1000000, 1000000]")
    if arguments["denominator"] == 0:
        return tool_error("denominator must not be zero")
    structured = {"quotient": arguments["numerator"] / arguments["denominator"]}
    return {
        "content": [{"type": "text", "text": json.dumps(structured, allow_nan=False)}],
        "structuredContent": structured,
        "isError": False,
    }


def simulate_request(request: object) -> dict[str, Any]:
    """Handle tools/list and tools/call fixtures after an ASSUMED initialization."""
    if not isinstance(request, dict):
        return protocol_error(None, -32600, "expected a request object")
    request_id = request.get("id")
    valid_id = type(request_id) in (int, str)
    if not valid_id or request.get("jsonrpc") != "2.0":
        return protocol_error(request_id if valid_id else None, -32600, "invalid request")
    method = request.get("method")
    if not isinstance(method, str):
        return protocol_error(request_id, -32600, "method must be a string")
    params = request.get("params", {})
    if not isinstance(params, dict):
        return protocol_error(request_id, -32602, "params must be an object")
    if method == "tools/list":
        if params:
            return protocol_error(request_id, -32602, "pagination is not implemented")
        # Copy metadata so a caller cannot mutate the shared fixture.
        result = {"tools": [json.loads(json.dumps(TOOL))]}
    elif method == "tools/call":
        if set(params) - {"name", "arguments"} or params.get("name") != TOOL["name"]:
            return protocol_error(request_id, -32602, "unknown tool or unsupported parameters")
        result = divide(params.get("arguments", {}))
    else:
        return protocol_error(request_id, -32601, "method not implemented in this exercise")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> int:
    fixtures = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "divide", "arguments": {"numerator": 7, "denominator": 2},
        }},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "divide", "arguments": {"numerator": 7, "denominator": 0},
        }},
    ]
    for request in fixtures:
        print(json.dumps(simulate_request(request), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
