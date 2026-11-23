"""Deterministic synthetic experiment demonstrating adapter fitting and merging."""

from __future__ import annotations

import numpy as np

from lora_adaptation_lab.core import fit_adapter, mean_squared_error


def run_demo(seed: int = 17, steps: int = 600) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    train_x = rng.normal(size=(512, 12))
    test_x = rng.normal(size=(256, 12))
    base = rng.normal(scale=0.15, size=(12, 8))
    truth_a = rng.normal(size=(12, 2))
    truth_b = rng.normal(size=(2, 8))
    true_delta = 0.5 * (truth_a @ truth_b)
    train_y = train_x @ true_delta

    adapter = fit_adapter(train_x, train_y, rank=2, alpha=1.0, steps=steps, seed=seed)
    predictions = adapter.apply(test_x, base)
    target = test_x @ (base + true_delta)
    merged = test_x @ adapter.merge(base)
    return {
        "rank": adapter.rank,
        "trainable_parameters": adapter.trainable_parameters,
        "full_weight_parameters": int(base.size),
        "parameter_ratio": adapter.trainable_parameters / base.size,
        "test_mse": mean_squared_error(predictions, target),
        "merge_max_abs_difference": float(np.max(np.abs(predictions - merged))),
    }
