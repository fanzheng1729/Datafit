"""Shared helpers for rank-factor optimizer entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rank_optimization_model import (
    Evaluation,
    RankOptimizationModel,
    VariableDict,
    add_scaled_variables,
    copy_variables,
    variable_norm,
)


MIN_OBJECTIVE_DECREASE = 1.0e-10
GAUGE_TOLERANCE = 1.0e-8
INITIAL_STEP_SCALE = 3.0e-16
STEP_MULTIPLIERS = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]


def negative_gradient(gradient: VariableDict) -> VariableDict:
    """Return the raw steepest-descent direction."""

    out: VariableDict = {}
    for key, value in gradient.items():
        out[key] = -value if isinstance(value, float) else -value.copy()
    return out


def normalize_direction(direction: VariableDict) -> tuple[VariableDict, float]:
    """Normalize a direction in the current Euclidean coefficient metric."""

    norm = variable_norm(direction)
    if norm == 0.0:
        return copy_variables(direction), norm
    out: VariableDict = {}
    for key, value in direction.items():
        out[key] = value / norm if isinstance(value, np.ndarray) else float(value) / norm
    return out, norm


def max_abs(values: dict[str, float]) -> float:
    return max(abs(value) for value in values.values())


def residual_rms_summary(model: RankOptimizationModel, evaluation: Evaluation) -> dict[str, float]:
    """Compact residual magnitudes used in console and JSON output."""

    return model.residual_rms_from_residuals(evaluation.residuals)


def candidate_summary_with_evaluation(
    model: RankOptimizationModel,
    base_variables: VariableDict,
    direction: VariableDict,
    step: float,
    base_objective: float,
) -> tuple[dict[str, Any], VariableDict, Evaluation]:
    """Evaluate, retract, and classify one line-search candidate."""

    candidate = add_scaled_variables(base_variables, direction, step)
    candidate = model.retract_variables(candidate)
    evaluation = model.evaluate(candidate)
    objective = model.objective_from_residuals(evaluation.residuals)
    gauge_errors = model.gauge_errors_from_fields(evaluation.fields)
    objective_change = objective - base_objective
    accepted = (
        objective_change < -MIN_OBJECTIVE_DECREASE
        and max_abs(gauge_errors) <= GAUGE_TOLERANCE
    )
    return (
        {
            "step": float(step),
            "objective": float(objective),
            "objective_change": float(objective_change),
            "accepted": bool(accepted),
            "gauge_errors": gauge_errors,
            "max_abs_gauge_error": float(max_abs(gauge_errors)),
        },
        candidate,
        evaluation,
    )


def save_state(path: Path, variables: VariableDict) -> None:
    """Persist a mixed variable dictionary to a compressed NPZ file."""

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, float] = {}
    for key, value in variables.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            scalars[key] = float(value)
    arrays["_scalar_names"] = np.array(list(scalars), dtype=object)
    arrays["_scalar_values"] = np.array([scalars[key] for key in scalars], dtype=float)
    np.savez_compressed(path, **arrays)


def history_row(
    iteration: int,
    objective: float,
    objective_change: float,
    accepted_step: float,
    gradient_norm: float,
    projected_gradient_norm: float,
    directional_derivative: float,
    gauge_errors: dict[str, float],
    residuals: dict[str, float],
) -> dict[str, float | int]:
    """Build one CSV row for an accepted optimizer step."""

    return {
        "iteration": int(iteration),
        "objective": float(objective),
        "objective_change": float(objective_change),
        "accepted_step": float(accepted_step),
        "gradient_norm": float(gradient_norm),
        "projected_gradient_norm": float(projected_gradient_norm),
        "directional_derivative": float(directional_derivative),
        "max_abs_gauge_error": float(max_abs(gauge_errors)),
        "omega_x1_gauge_error": float(gauge_errors["omega_x1_00"]),
        "theta_x1x1_gauge_error": float(gauge_errors["theta_x1x1_00"]),
        "fomega_rms": float(residuals["fomega"]),
        "fzeta_rms": float(residuals["fzeta"]),
        "divergence_rms": float(residuals["divergence"]),
        "curl_rms": float(residuals["curl"]),
    }


def write_history_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write accepted-step history with stable column order."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_best_candidate(
    candidates: list[dict[str, Any]],
    candidate_variables: list[VariableDict],
) -> tuple[int | None, VariableDict | None]:
    """Pick the accepted candidate with the lowest objective."""

    passing = [index for index, item in enumerate(candidates) if item["accepted"]]
    if not passing:
        return None, None
    best_index = min(passing, key=lambda index: candidates[index]["objective"])
    return best_index, candidate_variables[best_index]


def output_paths(prefix: str) -> dict[str, Path]:
    """Resolve result/history/state file names next to this module."""

    stem = Path(prefix).name
    return {
        "results": Path(__file__).with_name(f"{stem}_results.json"),
        "history": Path(__file__).with_name(f"{stem}_history.csv"),
        "state": Path(__file__).with_name(f"{stem}_state.npz"),
    }
