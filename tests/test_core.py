from __future__ import annotations

import unittest

import numpy as np

from lora_adaptation_lab import LowRankAdapter, fit_adapter, mean_squared_error
from lora_adaptation_lab.demo import run_demo


class AdapterTests(unittest.TestCase):
    def test_apply_and_merge_are_equivalent(self) -> None:
        adapter = LowRankAdapter(
            np.array([[1.0], [2.0]]), np.array([[0.5, -1.0]]), alpha=2.0
        )
        x = np.array([[2.0, 3.0], [-1.0, 4.0]])
        base = np.array([[0.2, 0.4], [0.1, -0.3]])
        np.testing.assert_allclose(adapter.apply(x, base), x @ adapter.merge(base))

    def test_trainable_parameter_count_is_low_rank(self) -> None:
        adapter = LowRankAdapter(np.ones((10, 2)), np.ones((2, 8)), alpha=2.0)
        self.assertEqual(adapter.trainable_parameters, 36)
        self.assertLess(adapter.trainable_parameters, 10 * 8)

    def test_fitting_reduces_error_on_held_out_synthetic_data(self) -> None:
        rng = np.random.default_rng(23)
        train_x = rng.normal(size=(400, 10))
        test_x = rng.normal(size=(200, 10))
        left = rng.normal(size=(10, 2))
        right = rng.normal(size=(2, 6))
        train_y = train_x @ (0.5 * left @ right)
        test_y = test_x @ (0.5 * left @ right)
        adapter = fit_adapter(train_x, train_y, rank=2, alpha=1.0, steps=800, seed=3)
        error = mean_squared_error(adapter.apply(test_x, np.zeros((10, 6))), test_y)
        self.assertLess(error, 0.02)

    def test_demo_is_reproducible_and_merge_is_exact(self) -> None:
        first = run_demo(seed=11, steps=300)
        second = run_demo(seed=11, steps=300)
        self.assertEqual(first, second)
        self.assertEqual(first["merge_max_abs_difference"], 0.0)
        self.assertLess(first["parameter_ratio"], 0.6)

    def test_rejects_invalid_dimensions_and_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            LowRankAdapter(np.ones((2, 1)), np.ones((2, 3)), alpha=1.0)
        with self.assertRaises(ValueError):
            fit_adapter(np.array([[np.nan, 1.0]]), np.ones((1, 2)), rank=1)

    def test_rejects_mismatched_targets_and_invalid_rank(self) -> None:
        with self.assertRaises(ValueError):
            fit_adapter(np.ones((3, 2)), np.ones((2, 1)), rank=1)
        with self.assertRaises(ValueError):
            fit_adapter(np.ones((3, 2)), np.ones((3, 1)), rank=3)


if __name__ == "__main__":
    unittest.main()
