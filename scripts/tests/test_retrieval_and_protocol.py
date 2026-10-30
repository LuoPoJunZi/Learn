"""Offline regression tests for RRF ranking and MCP tool message exercises."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType


EXAMPLES = Path(__file__).resolve().parents[2] / "Python" / "Modern AI" / "examples"


def load_example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rrf = load_example("reciprocal_rank_fusion")
mcp = load_example("mcp_tool_messages")


class FusionTests(unittest.TestCase):
    def test_formula_and_one_based_rank(self) -> None:
        result = dict(rrf.reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
        self.assertAlmostEqual(result["a"], 1 / 61)
        self.assertAlmostEqual(result["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(result["c"], 1 / 62)

    def test_weight_and_zero_weight(self) -> None:
        result = rrf.reciprocal_rank_fusion([["a"], ["b"]], weights=[0, 2])
        self.assertEqual(result[0][0], "b")
        self.assertAlmostEqual(result[0][1], 2 / 61)
        self.assertEqual(len(result), 1)

    def test_ties_and_empty_results(self) -> None:
        self.assertEqual(rrf.reciprocal_rank_fusion([]), [])
        self.assertEqual(rrf.reciprocal_rank_fusion([[], []]), [])
        result = rrf.reciprocal_rank_fusion([["z"], ["a"]])
        self.assertEqual([item[0] for item in result], ["a", "z"])

    def test_reject_duplicates_and_invalid_configuration(self) -> None:
        for rankings, kwargs in [
            ([["a", "a"]], {}), ([[""]], {}),
            ([["a"]], {"k": 0}), ([["a"]], {"k": float("nan")}),
            ([["a"]], {"weights": []}), ([["a"]], {"weights": [-1]}),
            ([["a"]], {"weights": [float("inf")]}),
        ]:
            with self.subTest(rankings=rankings, kwargs=kwargs), self.assertRaises(ValueError):
                rrf.reciprocal_rank_fusion(rankings, **kwargs)

    def test_recall_uses_unique_ids_and_explicit_labels(self) -> None:
        self.assertEqual(rrf.recall_at_k(["a", "a", "b"], {"a", "b"}, 2), 0.5)
        with self.assertRaises(ValueError):
            rrf.recall_at_k([], set(), 2)
        with self.assertRaises(ValueError):
            rrf.recall_at_k(["a"], {"a"}, 0)


class MessageTests(unittest.TestCase):
    def call(self, arguments: object, name: str = "divide") -> dict:
        return mcp.simulate_request({
            "jsonrpc": "2.0", "id": "req-7", "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })

    def test_discovery_and_metadata_copy(self) -> None:
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        response = mcp.simulate_request(request)
        self.assertEqual(response["id"], 1)
        tool = response["result"]["tools"][0]
        self.assertIn("inputSchema", tool)
        self.assertIn("outputSchema", tool)
        tool["name"] = "mutated"
        self.assertEqual(mcp.simulate_request(request)["result"]["tools"][0]["name"], "divide")

    def test_success_and_text_structured_consistency(self) -> None:
        response = self.call({"numerator": 7, "denominator": 2})
        self.assertEqual(response["id"], "req-7")
        result = response["result"]
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"], {"quotient": 3.5})
        self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])

    def test_tool_errors_do_not_become_protocol_errors(self) -> None:
        for arguments in [
            {}, {"numerator": 7, "denominator": 0},
            {"numerator": True, "denominator": 2},
            {"numerator": float("nan"), "denominator": 2},
            {"numerator": 1.5, "denominator": 2},
            {"numerator": 1000001, "denominator": 2},
            {"numerator": 1, "denominator": 2, "extra": 3}, [],
        ]:
            with self.subTest(arguments=arguments):
                response = self.call(arguments)
                self.assertNotIn("error", response)
                self.assertTrue(response["result"]["isError"])

    def test_integer_valued_json_number(self) -> None:
        self.assertFalse(self.call({"numerator": 6.0, "denominator": 2})["result"]["isError"])

    def test_protocol_errors(self) -> None:
        self.assertEqual(self.call({}, "unknown")["error"]["code"], -32602)
        self.assertEqual(mcp.simulate_request(None)["error"]["code"], -32600)
        response = mcp.simulate_request({"jsonrpc": "2.0", "id": 1, "method": "unknown"})
        self.assertEqual(response["error"]["code"], -32601)
        response = mcp.simulate_request({"jsonrpc": "2.0", "id": True, "method": "tools/list"})
        self.assertIsNone(response["id"])


if __name__ == "__main__":
    unittest.main()
