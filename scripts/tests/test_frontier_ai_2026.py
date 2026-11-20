"""Offline regression tests for MLA, GRPO, and A2A protocol teaching examples."""

from __future__ import annotations

import importlib.util
import math
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


mla = load_example("multi_head_latent_attention")
grpo = load_example("grpo_group_relative_policy")
a2a = load_example("a2a_agent_protocol")


class MLATests(unittest.TestCase):
    def test_causal_mask_and_output_shapes(self) -> None:
        config, weights, tokens = mla.build_demo_setup()
        outputs, latent_cache, attn_weights = mla.mla_causal_forward(tokens, weights, config)
        self.assertEqual(len(latent_cache), len(tokens))
        self.assertEqual(len(latent_cache[0]), config.kv_latent_dim)
        self.assertEqual(len(outputs), len(tokens))
        self.assertEqual(len(outputs[0]), config.num_heads * config.head_dim)
        # Verify strict causal masking and row normalization on every head.
        for head_attn in attn_weights:
            for pos, row in enumerate(head_attn):
                self.assertAlmostEqual(sum(row), 1.0, places=6)
                for future_pos in range(pos + 1, len(row)):
                    self.assertEqual(row[future_pos], 0.0)

    def test_cache_compression_and_validation(self) -> None:
        counts = mla.compare_kv_cache_scalars(4096, 32, 128, 8, 512)
        self.assertEqual(counts["MHA"] // counts["MLA"], 16)
        self.assertEqual(counts["GQA"] // counts["MLA"], 4)
        with self.assertRaises(ValueError):
            mla.MLAConfig(d_model=6, num_heads=2, head_dim=2, kv_latent_dim=8)
        with self.assertRaises(ValueError):
            mla.compare_kv_cache_scalars(10, 6, 4, 4, 2)
        config, weights, _ = mla.build_demo_setup()
        with self.assertRaises(ValueError):
            mla.mla_causal_forward([[float("nan")] * config.d_model], weights, config)


class GRPOTests(unittest.TestCase):
    def test_advantages_and_zero_variance(self) -> None:
        adv = grpo.compute_group_advantages([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)
        self.assertGreater(adv[0], 0.0)
        self.assertLess(adv[1], 0.0)
        self.assertEqual(grpo.compute_group_advantages([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_schulman_kl_is_non_negative(self) -> None:
        self.assertAlmostEqual(grpo.schulman_kl_approx(-1.2, -1.2), 0.0, places=9)
        self.assertGreater(grpo.schulman_kl_approx(-1.5, -0.8), 0.0)
        self.assertGreater(grpo.schulman_kl_approx(-0.4, -1.3), 0.0)

    def test_toy_training_shifts_probability_to_high_reward(self) -> None:
        p0, p1, history = grpo.run_toy_grpo_training(steps=10, lr=0.4)
        self.assertAlmostEqual(sum(p0), 1.0, places=6)
        self.assertAlmostEqual(sum(p1), 1.0, places=6)
        # Candidates 0 and 3 have reward 1.0; candidate 2 has reward 0.0.
        self.assertGreater(p1[0] + p1[3], p0[0] + p0[3])
        self.assertLess(p1[2], p0[2])
        self.assertGreater(history[-1], history[0])

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            grpo.compute_group_advantages([])
        with self.assertRaises(ValueError):
            grpo.compute_group_advantages([math.inf, 1.0])
        with self.assertRaises(ValueError):
            grpo.run_toy_grpo_training(steps=0)


class A2ATests(unittest.TestCase):
    def test_agent_card_defensive_copy(self) -> None:
        agent = a2a.OfflineA2AAgent()
        resp1 = agent.handle_request({"jsonrpc": "2.0", "id": 1, "method": "agent/card"})
        resp1["result"]["skills"].clear()
        resp2 = agent.handle_request({"jsonrpc": "2.0", "id": 2, "method": "agent/card"})
        self.assertEqual(len(resp2["result"]["skills"]), 2)

    def test_task_lifecycle_and_error_layers(self) -> None:
        agent = a2a.OfflineA2AAgent()
        ok = agent.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tasks/send",
                "params": {"id": "t1", "skillId": "word-count", "text": "one two three"},
            }
        )
        self.assertEqual(ok["result"]["status"]["state"], "completed")
        self.assertEqual(ok["result"]["artifacts"][0]["data"]["wordCount"], 3)

        # Duplicate task ID rejected at protocol level.
        dup = agent.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tasks/send",
                "params": {"id": "t1", "skillId": "word-count", "text": "again"},
            }
        )
        self.assertEqual(dup["error"]["code"], -32602)

        # Terminal task cannot be canceled.
        cancel_done = agent.handle_request(
            {"jsonrpc": "2.0", "id": 3, "method": "tasks/cancel", "params": {"id": "t1"}}
        )
        self.assertEqual(cancel_done["error"]["code"], -32002)

        # Deferred task can be canceled cleanly.
        agent.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tasks/send",
                "params": {"id": "t2", "skillId": "extract-title", "text": "Title", "defer": True},
            }
        )
        canceled = agent.handle_request(
            {"jsonrpc": "2.0", "id": 5, "method": "tasks/cancel", "params": {"id": "t2"}}
        )
        self.assertEqual(canceled["result"]["status"]["state"], "canceled")
        self.assertEqual(canceled["result"]["stateHistory"], ["submitted", "working", "canceled"])

        # Unknown task / method / empty text failure path.
        missing = agent.handle_request(
            {"jsonrpc": "2.0", "id": 6, "method": "tasks/get", "params": {"id": "nope"}}
        )
        self.assertEqual(missing["error"]["code"], -32001)

        bad_method = agent.handle_request({"jsonrpc": "2.0", "id": 7, "method": "tasks/unknown"})
        self.assertEqual(bad_method["error"]["code"], -32601)

        failed = agent.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tasks/send",
                "params": {"id": "t3", "skillId": "extract-title", "text": "   "},
            }
        )
        self.assertEqual(failed["result"]["status"]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
