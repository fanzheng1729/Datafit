#!/usr/bin/env python3
"""Check retraction/refit using the shared explicit-S optimization model.

The optimizer relies on retraction after every trial point.  This script
deliberately rescales equivalent left/right factors, retracts them, refits them
to B-spline coefficients, and verifies that the represented field is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np

from rank_factor_tools import retract_and_refit, synthesize, trapezoid_weights
from rank_optimization_model import DATA_PATH, RankCore, RankOptimizationModel, VariableDict, relative_norm


RESULTS_PATH = Path(__file__).with_name("retraction_check_results.json")


def evaluated_core(core: RankCore, variables: VariableDict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluated left/right factors and singular values for one core."""

    p = variables[core.p_key]
    q = variables[core.q_key]
    s = variables[core.s_key]
    assert isinstance(p, np.ndarray)
    assert isinstance(q, np.ndarray)
    assert isinstance(s, np.ndarray)
    return core.x0 @ p, core.y0 @ q, s


def check_core(
    core: RankCore,
    variables: VariableDict,
    left_weights: np.ndarray | None,
    right_weights: np.ndarray | None,
) -> dict[str, Any]:
    """Stress one core by applying a harmless column-rescaling symmetry."""

    left, right, singular_values = evaluated_core(core, variables)
    baseline_field = synthesize(left, right, singular_values)

    rank = singular_values.size
    # Multiplying left columns and dividing right columns by the same values
    # leaves the represented matrix unchanged but moves far from the normalized
    # explicit-S chart.
    scaling = np.geomspace(0.25, 4.0, rank)
    scaled_left = left * scaling[None, :]
    scaled_right = right / scaling[None, :]
    scaled_field = synthesize(scaled_left, scaled_right, singular_values)

    result = retract_and_refit(
        scaled_left,
        scaled_right,
        singular_values,
        core.x0,
        core.y0,
        core.parity,
        left_weights=left_weights,
        right_weights=right_weights,
    )
    retracted_field = synthesize(result.left, result.right, result.singular_values)
    refit_left = core.x0 @ result.left_coefficients
    refit_right = core.y0 @ result.right_coefficients
    refit_field = synthesize(refit_left, refit_right, result.singular_values)

    return {
        "rank": int(rank),
        "scaled_field_relative_error": relative_norm(scaled_field - baseline_field, baseline_field),
        "retracted_field_relative_error": relative_norm(retracted_field - baseline_field, baseline_field),
        "refit_field_relative_error": relative_norm(refit_field - baseline_field, baseline_field),
        "left_gram_error": result.left_gram_error,
        "right_gram_error": result.right_gram_error,
        "left_refit_relative_error": result.left_refit_relative_error,
        "right_refit_relative_error": result.right_refit_relative_error,
        "largest_singular_value": float(result.singular_values[0]),
        "smallest_singular_value": float(result.singular_values[-1]),
    }


def summarize_checks(
    model: RankOptimizationModel,
    variables: VariableDict,
    left_weights: np.ndarray | None,
    right_weights: np.ndarray | None,
    thresholds: dict[str, float],
) -> tuple[bool, dict[str, Any]]:
    """Run the retraction/refit check for every optimized core."""

    core_results = {
        core.name: check_core(core, variables, left_weights, right_weights)
        for core in model.cores
    }
    passed = True
    for result in core_results.values():
        passed = passed and result["retracted_field_relative_error"] < thresholds["field_relative_error"]
        passed = passed and result["refit_field_relative_error"] < thresholds["field_relative_error"]
        passed = passed and result["left_gram_error"] < thresholds["gram_error"]
        passed = passed and result["right_gram_error"] < thresholds["gram_error"]
        passed = passed and result["left_refit_relative_error"] < thresholds["refit_relative_error"]
        passed = passed and result["right_refit_relative_error"] < thresholds["refit_relative_error"]
    return bool(passed), core_results


def main() -> int:
    """Run primary and trapezoid-weight stress checks."""

    model = RankOptimizationModel(DATA_PATH)
    variables = model.variables
    x1_trapezoid_weights = trapezoid_weights(model.x1)
    x2_trapezoid_weights = trapezoid_weights(model.x2)
    thresholds = {
        "field_relative_error": 1.0e-10,
        "gram_error": 1.0e-10,
        "refit_relative_error": 1.0e-10,
    }
    # Identity weights match the current optimizer chart.  The trapezoid pass is
    # retained as a stress check because the adaptive grid has an enormous
    # dynamic range in cell sizes.
    passed, core_results = summarize_checks(model, variables, None, None, thresholds)
    trapezoid_passed, trapezoid_results = summarize_checks(
        model, variables, x1_trapezoid_weights, x2_trapezoid_weights, thresholds
    )

    output = {
        "description": "Explicit-S retraction and B-spline refit check using rank_optimization_model.py.",
        "passed": bool(passed),
        "primary_weight_rule": "identity weights, matching the current discrete SVD/grid RMS norm",
        "thresholds": thresholds,
        "cores": core_results,
        "stress_check": {
            "weight_rule": "raw one-dimensional trapezoid weights on x1 and x2 meshes",
            "passed": bool(trapezoid_passed),
            "reason_not_primary": (
                "The adaptive mesh quadrature weights span about 1e15, making the "
                "small retraction SVD nearly condition-limited in ordinary grid norm."
            ),
            "cores": trapezoid_results,
        },
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("Explicit-S retraction/refit check")
    print(f"  passed: {passed}")
    for name, result in core_results.items():
        print(
            f"  {name:5s} rank={result['rank']:3d} "
            f"retract={result['retracted_field_relative_error']:.3e} "
            f"refit={result['refit_field_relative_error']:.3e} "
            f"gram=({result['left_gram_error']:.3e}, {result['right_gram_error']:.3e})"
        )
    print(f"  raw trapezoid stress passed: {trapezoid_passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
