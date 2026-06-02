"""Utilities for explicit-S low-rank factor charts.

The future optimizer keeps a diagonal singular-amplitude vector ``s`` separate
from the left and right B-spline factors.  This module contains the mechanical
pieces needed after an accepted trial step:

1. retract evaluated factors to a weighted SVD-like normalization, and
2. refit the normalized evaluated factors back to B-spline coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetractionRefit:
    left: np.ndarray
    right: np.ndarray
    singular_values: np.ndarray
    left_coefficients: np.ndarray
    right_coefficients: np.ndarray
    left_gram_error: float
    right_gram_error: float
    left_refit_relative_error: float
    right_refit_relative_error: float


def synthesize(left: np.ndarray, right: np.ndarray, singular_values: np.ndarray) -> np.ndarray:
    """Return ``left @ diag(singular_values) @ right.T`` without forming diag."""

    return (left * singular_values[None, :]) @ right.T


def synthesize_triplet(
    left: np.ndarray,
    left_x1: np.ndarray,
    right: np.ndarray,
    right_x2: np.ndarray,
    singular_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return value, x1, and x2 low-rank syntheses with shared scaling.

    The three fields are
    ``left @ diag(s) @ right.T``,
    ``left_x1 @ diag(s) @ right.T``, and
    ``left @ diag(s) @ right_x2.T``.  Scaling ``left`` by ``s`` is shared
    between the value and x2 derivative, and the value/x1 products against
    ``right.T`` are batched in one matrix multiply.
    """

    weighted_left = left * singular_values[None, :]
    weighted_left_x1 = left_x1 * singular_values[None, :]
    value_and_x1 = np.vstack((weighted_left, weighted_left_x1)) @ right.T
    split = left.shape[0]
    value = value_and_x1[:split]
    x1 = value_and_x1[split:]
    x2 = weighted_left @ right_x2.T
    return value, x1, x2


def weighted_gram(factors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return ``factors.T @ W @ factors`` for diagonal weights."""

    weights = _weights_or_ones(factors.shape[0], weights)
    return factors.T @ (weights[:, None] * factors)


def trapezoid_weights(mesh: np.ndarray) -> np.ndarray:
    """Return positive trapezoid quadrature weights for a one-dimensional mesh."""

    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 1:
        raise ValueError("mesh must be one-dimensional")
    if mesh.size < 2:
        raise ValueError("mesh must contain at least two points")
    if np.any(np.diff(mesh) <= 0.0):
        raise ValueError("mesh must be strictly increasing")

    weights = np.empty_like(mesh)
    weights[0] = 0.5 * (mesh[1] - mesh[0])
    weights[-1] = 0.5 * (mesh[-1] - mesh[-2])
    weights[1:-1] = 0.5 * (mesh[2:] - mesh[:-2])
    return weights


def weighted_orthonormality_error(
    factors: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Frobenius error from weighted orthonormal columns."""

    gram = weighted_gram(factors, weights)
    return float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro"))


def retract_factors(
    left: np.ndarray,
    right: np.ndarray,
    singular_values: np.ndarray,
    left_weights: np.ndarray | None = None,
    right_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retract a low-rank chart to weighted-orthonormal factors.

    If ``L S M.T`` is the incoming field representation, the returned
    ``L_new, s_new, M_new`` satisfies the same field up to roundoff and
    ``L_new.T W_x L_new = I``, ``M_new.T W_y M_new = I``.
    """

    _check_factor_shapes(left, right, singular_values)
    wx = _weights_or_ones(left.shape[0], left_weights)
    wy = _weights_or_ones(right.shape[0], right_weights)

    sqrt_wx = np.sqrt(wx)
    sqrt_wy = np.sqrt(wy)
    q_left_weighted, r_left = np.linalg.qr(sqrt_wx[:, None] * left, mode="reduced")
    q_right_weighted, r_right = np.linalg.qr(sqrt_wy[:, None] * right, mode="reduced")

    left_basis = q_left_weighted / sqrt_wx[:, None]
    right_basis = q_right_weighted / sqrt_wy[:, None]
    core = (r_left * singular_values[None, :]) @ r_right.T
    core_left, new_singular_values, core_right_t = np.linalg.svd(core, full_matrices=False)

    new_left = left_basis @ core_left
    new_right = right_basis @ core_right_t.T
    _fix_right_factor_signs(new_left, new_right)
    return new_left, new_right, new_singular_values


def refit_left_coefficients(
    x_basis: np.ndarray,
    left: np.ndarray,
    parity: int,
) -> np.ndarray:
    """Solve B-spline coefficients for evaluated left factors.

    For odd reflected variables, the first row is the enforced origin row and is
    excluded exactly as in ``runfit.SVDfit``.
    """

    if parity == 1:
        return _solve_or_lstsq(x_basis[1:], left[1:])
    return _solve_or_lstsq(x_basis, left)


def refit_right_coefficients(y_basis: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Solve B-spline coefficients for evaluated right factors."""

    return _solve_or_lstsq(y_basis, right)


def retract_and_refit(
    left: np.ndarray,
    right: np.ndarray,
    singular_values: np.ndarray,
    x_basis: np.ndarray,
    y_basis: np.ndarray,
    parity: int,
    left_weights: np.ndarray | None = None,
    right_weights: np.ndarray | None = None,
) -> RetractionRefit:
    """Retract evaluated factors and refit them to B-spline coefficients."""

    new_left, new_right, new_singular_values = retract_factors(
        left, right, singular_values, left_weights, right_weights
    )
    p = refit_left_coefficients(x_basis, new_left, parity)
    q = refit_right_coefficients(y_basis, new_right)
    refit_left = x_basis @ p
    refit_right = y_basis @ q

    return RetractionRefit(
        left=new_left,
        right=new_right,
        singular_values=new_singular_values,
        left_coefficients=p,
        right_coefficients=q,
        left_gram_error=weighted_orthonormality_error(new_left, left_weights),
        right_gram_error=weighted_orthonormality_error(new_right, right_weights),
        left_refit_relative_error=_relative_norm(refit_left - new_left, new_left),
        right_refit_relative_error=_relative_norm(refit_right - new_right, new_right),
    )


def _weights_or_ones(size: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=float)
    out = np.asarray(weights, dtype=float)
    if out.shape != (size,):
        raise ValueError(f"expected weights of shape {(size,)}, got {out.shape}")
    if np.any(out <= 0.0):
        raise ValueError("weights must be positive")
    return out


def _check_factor_shapes(
    left: np.ndarray, right: np.ndarray, singular_values: np.ndarray
) -> None:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right factors must be two-dimensional")
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")
    if left.shape[1] != right.shape[1] or left.shape[1] != singular_values.size:
        raise ValueError(
            "left, right, and singular_values must have the same rank dimension"
        )


def _fix_right_factor_signs(left: np.ndarray, right: np.ndarray) -> None:
    for col in range(right.shape[1]):
        pivot = int(np.argmax(np.abs(right[:, col])))
        if right[pivot, col] < 0.0:
            left[:, col] *= -1.0
            right[:, col] *= -1.0


def _solve_or_lstsq(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if system.shape[0] == system.shape[1]:
        return np.linalg.solve(system, rhs)
    return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _relative_norm(error: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(error) / denominator)
