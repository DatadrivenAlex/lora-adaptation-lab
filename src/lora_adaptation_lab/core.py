"""Small, transparent NumPy implementation of low-rank residual adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _matrix(name: str, value: NDArray[np.floating]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class LowRankAdapter:
    """A trainable update ``scale * A @ B`` for a frozen linear weight matrix."""

    A: FloatArray
    B: FloatArray
    alpha: float

    def __post_init__(self) -> None:
        a = _matrix("A", self.A)
        b = _matrix("B", self.B)
        if a.shape[1] != b.shape[0]:
            raise ValueError("A's rank dimension must equal B's rank dimension")
        if min(a.shape) < 1 or min(b.shape) < 1:
            raise ValueError("adapter matrices must not be empty")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)

    @property
    def rank(self) -> int:
        return self.A.shape[1]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def trainable_parameters(self) -> int:
        return self.A.size + self.B.size

    def delta(self) -> FloatArray:
        return self.scale * (self.A @ self.B)

    def apply(self, inputs: NDArray[np.floating], base_weight: NDArray[np.floating]) -> FloatArray:
        x = _matrix("inputs", inputs)
        weight = _matrix("base_weight", base_weight)
        if weight.shape[0] != self.A.shape[0] or weight.shape[1] != self.B.shape[1]:
            raise ValueError("base_weight dimensions must match the adapter")
        if x.shape[1] != weight.shape[0]:
            raise ValueError("input width must match base_weight rows")
        return x @ (weight + self.delta())

    def merge(self, base_weight: NDArray[np.floating]) -> FloatArray:
        weight = _matrix("base_weight", base_weight)
        if weight.shape != (self.A.shape[0], self.B.shape[1]):
            raise ValueError("base_weight dimensions must match the adapter")
        return weight + self.delta()


def fit_adapter(
    inputs: NDArray[np.floating],
    residual_targets: NDArray[np.floating],
    *,
    rank: int,
    alpha: float | None = None,
    learning_rate: float = 0.05,
    steps: int = 400,
    l2: float = 0.0,
    seed: int = 7,
) -> LowRankAdapter:
    """Fit a low-rank adapter to residual targets using deterministic full-batch SGD."""
    x = _matrix("inputs", inputs)
    y = _matrix("residual_targets", residual_targets)
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("inputs and targets must have the same nonzero row count")
    if rank < 1 or rank > min(x.shape[1], y.shape[1]):
        raise ValueError("rank must be between 1 and the smaller matrix dimension")
    alpha_value = float(rank if alpha is None else alpha)
    if not np.isfinite(alpha_value) or alpha_value <= 0:
        raise ValueError("alpha must be a finite positive number")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be a finite nonnegative number")

    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.1, size=(x.shape[1], rank))
    b = np.zeros((rank, y.shape[1]), dtype=np.float64)
    scale = alpha_value / rank
    n = x.shape[0]
    for _ in range(steps):
        error = x @ (scale * (a @ b)) - y
        grad_b = scale * (a.T @ x.T @ error) / n + l2 * b
        grad_a = scale * (x.T @ error @ b.T) / n + l2 * a
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return LowRankAdapter(a, b, alpha_value)


def mean_squared_error(actual: NDArray[np.floating], expected: NDArray[np.floating]) -> float:
    left = _matrix("actual", actual)
    right = _matrix("expected", expected)
    if left.shape != right.shape:
        raise ValueError("actual and expected must have the same shape")
    return float(np.mean(np.square(left - right)))
