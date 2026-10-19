"""Regression checks for offline modern AI teaching examples."""

from __future__ import annotations

import math
from pathlib import Path
import random
import runpy
import unittest


EXAMPLES = Path(__file__).resolve().parents[2] / "Python" / "Modern AI" / "examples"


class ModernAIExamplesTests(unittest.TestCase):
    """Exercise numerical invariants and malformed inputs without external effects."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = runpy.run_path(str(EXAMPLES / "ai_agent_tool_calling.py"))
        cls.dpo = runpy.run_path(str(EXAMPLES / "rlhf_to_dpo_alignment.py"))
        cls.ssm = runpy.run_path(str(EXAMPLES / "mamba_state_space_model.py"))
        cls.vlm = runpy.run_path(str(EXAMPLES / "vlm_vision_language_model.py"))
        cls.compute = runpy.run_path(str(EXAMPLES / "test_time_compute_scaling.py"))

    def test_calculator_allows_bounded_arithmetic(self) -> None:
        calculate = self.agent["_run_calculator"]
        self.assertTrue(calculate({"expression": "(3 + 5) * 12 - 4"}).endswith("92"))
        self.assertTrue(calculate({"expression": "-3 / 2"}).endswith("-1.5"))

    def test_calculator_rejects_unbounded_or_executable_syntax(self) -> None:
        for expression in ("9**999999999", "__import__(1)", "[1]*999999999", "1/0", "1e999"):
            with self.subTest(expression=expression):
                result = self.agent["_run_calculator"]({"expression": expression})
                self.assertFalse(result.startswith("\u8ba1\u7b97\u7ed3\u679c"))

    def test_dpo_normalizes_each_prompt(self) -> None:
        model = self.dpo["LogProbTable"]()
        model.set("prompt", "win", 1000.0)
        model.set("prompt", "lose", 999.0)
        total = sum(math.exp(model.log_prob("prompt", answer)) for answer in ("win", "lose"))
        self.assertAlmostEqual(total, 1.0)

    def test_dpo_does_not_cap_a_large_negative_margin(self) -> None:
        reference = self.dpo["LogProbTable"]()
        policy = self.dpo["LogProbTable"]()
        for answer in ("win", "lose"):
            reference.set("prompt", answer, 0.0)
        policy.set("prompt", "win", -1000.0)
        policy.set("prompt", "lose", 0.0)
        pair = self.dpo["PreferencePair"]("prompt", "win", "lose")
        loss = self.dpo["dpo_loss_single"](policy, reference, pair, beta=1.0)
        self.assertAlmostEqual(loss, 1000.0)

    def test_classical_ssm_decay_and_state_shape(self) -> None:
        model = self.ssm["ClassicalSSM"](1, [math.exp(-0.3)], [1.0], [1.0])
        state = [1.0]
        for _ in range(10):
            state, _ = model.step(state, 0.0)
        self.assertAlmostEqual(state[0], math.exp(-3.0))
        with self.assertRaises(ValueError):
            model.step([], 0.0)
        with self.assertRaises(ValueError):
            self.ssm["ClassicalSSM"](2, [1.0], [1.0], [1.0])

    def test_vlm_rejects_partial_patches_and_truncated_projection(self) -> None:
        patcher = self.vlm["extract_patches_from_image"]
        for image, size in (([], 2), ([[1.0, 2.0], [3.0]], 2), ([[1.0]], 2)):
            with self.subTest(image=image), self.assertRaises(ValueError):
                patcher(image, size)
        with self.assertRaises(ValueError):
            self.vlm["project_vision_to_llm_dim"]([1.0, 2.0], [[1.0]], [0.0])

    def test_vlm_patch_order(self) -> None:
        patches = self.vlm["extract_patches_from_image"]([[1.0, 2.0], [3.0, 4.0]], 2)
        self.assertEqual(patches[0].pixels, [1.0, 2.0, 3.0, 4.0])

    def test_generated_answer_agrees_with_last_step(self) -> None:
        for seed in range(20):
            candidate = self.compute["simulate_model_inference"](
                "fixed problem", -999.0, candidate_id=seed, rng=random.Random(seed),
            )
            self.assertEqual(candidate.final_answer, candidate.steps[-1].intermediate_result)

    def test_correction_recomputes_intermediate_values(self) -> None:
        step = self.compute["ReasoningStep"](1, "incorrect", -100.0, False)
        candidate = self.compute["CandidateSolution"](1, [step], -100.0)
        corrected = self.compute["self_correction_loop"](candidate)
        self.assertEqual([s.intermediate_result for s in corrected.steps], [24.0, 36.0, 32.0])
        self.assertEqual(corrected.final_answer, corrected.steps[-1].intermediate_result)

    def test_empty_vote_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.compute["self_consistency_voting"]([])


if __name__ == "__main__":
    unittest.main()
