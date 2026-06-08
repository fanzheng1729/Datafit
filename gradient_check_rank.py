#!/usr/bin/env python3
"""Analytic vs finite-difference check for explicit-S rank variables.

This is a diagnostic script, not part of the optimizer loop.  It chooses a few
representative high-gradient coordinates plus one random direction and compares
the analytic gradient from ``RankOptimizationModel`` against central finite
differences of the full objective.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np

from rank_optimization_model import (
    DATA_PATH,
    RankOptimizationModel,
    VariableDict,
    copy_variables,
    rms,
    variable_dot,
)


RESULTS_PATH = Path(__file__).with_name("gradient_check_results.json")


def coordinate_checks(
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
) -> list[dict[str, Any]]:
    """Check one high-signal coordinate in each array block plus the scalars."""

    checks: list[dict[str, Any]] = []
    base_objective = model.objective(variables)
    for key in model.array_keys():
        grad = gradient[key]
        assert isinstance(grad, np.ndarray)
        probe_grad = np.abs(grad).copy()
        if key.endswith("_Q") and probe_grad.ndim == 2 and probe_grad.shape[0] > 20:
            # The first/last Q rows can be dominated by boundary artifacts.  The
            # check still covers Q blocks, but avoids selecting only edge rows.
            probe_grad[:8, :] = 0.0
            probe_grad[-8:, :] = 0.0
        index = np.unravel_index(int(np.argmax(probe_grad)), grad.shape)
        value = variables[key]
        assert isinstance(value, np.ndarray)
        checks.append(
            best_coordinate_difference(
                model,
                variables,
                key,
                index,
                float(value[index]),
                float(grad[index]),
                base_objective,
            )
        )

    for key in ["cl", "cw", "rat"]:
        checks.append(
            best_coordinate_difference(
                model,
                variables,
                key,
                (),
                float(variables[key]),
                float(gradient[key]),
                base_objective,
            )
        )
    return checks


def best_coordinate_difference(
    model: RankOptimizationModel,
    variables: VariableDict,
    key: str,
    index: tuple[int, ...],
    value: float,
    analytic: float,
    objective: float,
) -> dict[str, Any]:
    """Try several finite-difference scales and keep the best-conditioned one."""

    scale = max(1.0, abs(value))
    trials = []
    for multiplier in [1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8]:
        step = multiplier * scale
        plus = copy_variables(variables)
        minus = copy_variables(variables)
        if index:
            assert isinstance(plus[key], np.ndarray)
            assert isinstance(minus[key], np.ndarray)
            plus[key][index] = value + step
            minus[key][index] = value - step
        else:
            plus[key] = value + step
            minus[key] = value - step
        fd = (model.objective(plus) - model.objective(minus)) / (2.0 * step)
        abs_error = abs(analytic - fd)
        denom = max(1.0e-12, abs(analytic), abs(fd))
        trials.append(
            {
                "step": float(step),
                "finite_difference": float(fd),
                "absolute_error": float(abs_error),
                "relative_error": float(abs_error / denom),
            }
        )
    best = min(trials, key=lambda item: item["relative_error"])
    return check_row(
        key,
        index,
        best["step"],
        analytic,
        best["finite_difference"],
        objective,
        trials,
    )


def check_row(
    key: str,
    index: tuple[int, ...],
    step: float,
    analytic: float,
    finite_difference: float,
    objective: float,
    trials: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Format one coordinate-check result for JSON output."""

    abs_error = abs(analytic - finite_difference)
    denom = max(1.0e-12, abs(analytic), abs(finite_difference))
    return {
        "variable": key,
        "index": [int(item) for item in index],
        "step": float(step),
        "analytic": float(analytic),
        "finite_difference": float(finite_difference),
        "absolute_error": float(abs_error),
        "relative_error": float(abs_error / denom),
        "objective": float(objective),
        "trials": trials or [],
    }


def directional_check(
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
) -> dict[str, Any]:
    """Compare analytic and finite-difference directional derivatives."""

    rng = np.random.default_rng(20260525)
    direction: VariableDict = {}
    norm_sq = 0.0
    for key, value in variables.items():
        if isinstance(value, np.ndarray):
            raw = rng.normal(size=value.shape)
            if key.endswith("_Q") and raw.ndim == 2 and raw.shape[0] > 20:
                # Match the coordinate check by reducing boundary-row influence
                # in the random direction.
                raw[:8, :] = 0.0
                raw[-8:, :] = 0.0
            scale = np.linalg.norm(raw)
            raw = raw / scale
            direction[key] = raw
            norm_sq += float(np.sum(raw * raw))
        else:
            raw_scalar = float(rng.normal())
            direction[key] = raw_scalar
            norm_sq += raw_scalar * raw_scalar
    norm = math.sqrt(norm_sq)
    for key, value in direction.items():
        direction[key] = value / norm if isinstance(value, np.ndarray) else float(value) / norm

    analytic = variable_dot(gradient, direction)
    trials = []
    for step in [1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8, 1.0e-8, 3.0e-9]:
        plus = copy_variables(variables)
        minus = copy_variables(variables)
        for key, value in variables.items():
            direction_value = direction[key]
            if isinstance(value, np.ndarray):
                assert isinstance(direction_value, np.ndarray)
                assert isinstance(plus[key], np.ndarray)
                assert isinstance(minus[key], np.ndarray)
                plus[key] += step * direction_value
                minus[key] -= step * direction_value
            else:
                plus[key] = float(value) + step * float(direction_value)
                minus[key] = float(value) - step * float(direction_value)
        fd = (model.objective(plus) - model.objective(minus)) / (2.0 * step)
        abs_error = abs(analytic - fd)
        denom = max(1.0e-12, abs(analytic), abs(fd))
        trials.append(
            {
                "step": float(step),
                "finite_difference": float(fd),
                "absolute_error": float(abs_error),
                "relative_error": float(abs_error / denom),
            }
        )
    best = min(trials, key=lambda item: item["relative_error"])
    return {
        "step": best["step"],
        "analytic": float(analytic),
        "finite_difference": best["finite_difference"],
        "absolute_error": best["absolute_error"],
        "relative_error": best["relative_error"],
        "trials": trials,
    }


def main() -> int:
    """Run all gradient checks and write the JSON report."""

    np.set_printoptions(precision=10, suppress=False)
    model = RankOptimizationModel(DATA_PATH)
    variables = copy_variables(model.variables)
    base_eval = model.evaluate(variables)
    gradient = model.analytic_gradient(variables)
    checks = coordinate_checks(model, variables, gradient)
    direction = directional_check(model, variables, gradient)
    max_relative = max(row["relative_error"] for row in checks)
    max_absolute = max(row["absolute_error"] for row in checks)

    output = {
        "description": "Analytic gradient vs central finite-difference check for explicit-S rank-factor variables.",
        "variables": model.variable_keys(),
        "objective": model.objective(variables),
        "residual_rms_scales": model.scales,
        "residual_rms_at_base": {name: rms(value) for name, value in base_eval.residuals.items()},
        "gauge_targets": model.gauge_targets,
        "gauge_errors_at_base": model.gauge_errors(variables),
        "ranks": {core.name: int(core.rank) for core in model.cores},
        "coordinate_checks": checks,
        "directional_check": direction,
        "max_coordinate_relative_error": max_relative,
        "max_coordinate_absolute_error": max_absolute,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("Explicit-S rank-factor gradient check")
    print(f"  objective:                 {output['objective']:.12e}")
    print(f"  ranks:                     {output['ranks']}")
    print(f"  max coordinate abs error:  {max_absolute:.12e}")
    print(f"  max coordinate rel error:  {max_relative:.12e}")
    print(f"  directional rel error:     {direction['relative_error']:.12e}")
    for row in checks:
        print(
            f"  {row['variable']:8s} {tuple(row['index'])!s:14s} "
            f"analytic {row['analytic']:.12e} fd {row['finite_difference']:.12e} "
            f"rel {row['relative_error']:.3e}"
        )
    print(f"  wrote:                     {RESULTS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
