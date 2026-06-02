#!/usr/bin/env python3
"""Compare constrained/retracted +gradient and -gradient trial steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np

from rank_optimization_model import (
    DATA_PATH,
    RankOptimizationModel,
    VariableDict,
    add_scaled_variables,
    copy_variables,
    variable_dot,
    variable_norm,
)


STATE_NPZ = Path(__file__).with_name("rank_optimization_state.npz")
RESULTS_PATH = Path(__file__).with_name("gradient_direction_sign_test.json")
GAUGE_TOLERANCE = 1.0e-8
STEPS = [
    3.0e-15,
    1.0e-15,
    3.0e-16,
    1.0e-16,
    3.0e-17,
    1.0e-17,
    3.0e-18,
    1.0e-18,
    3.0e-19,
    1.0e-19,
    3.0e-20,
    1.0e-20,
    3.0e-21,
    1.0e-21,
]


def load_state(path: Path) -> VariableDict:
    data = np.load(path, allow_pickle=True)
    variables: VariableDict = {}
    scalar_names = [str(item) for item in data["_scalar_names"]]
    scalar_values = data["_scalar_values"]
    for key in data.files:
        if key.startswith("_"):
            continue
        variables[key] = np.asarray(data[key], dtype=float)
    for key, value in zip(scalar_names, scalar_values):
        variables[key] = float(value)
    return variables


def max_abs(values: dict[str, float]) -> float:
    return max(abs(value) for value in values.values())


def normalize(direction: VariableDict) -> tuple[VariableDict, float]:
    norm = variable_norm(direction)
    out = copy_variables(direction)
    if norm == 0.0:
        return out, norm
    for key, value in out.items():
        out[key] = value / norm if isinstance(value, np.ndarray) else float(value) / norm
    return out, norm


def signed_direction(
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    sign: float,
) -> tuple[VariableDict, float]:
    raw: VariableDict = {}
    for key, value in gradient.items():
        raw[key] = sign * value if isinstance(value, float) else sign * value.copy()
    tangent = model.projected_tangent_direction(variables, raw)
    return normalize(tangent)


def trial_rows(
    model: RankOptimizationModel,
    variables: VariableDict,
    direction: VariableDict,
    objective: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in STEPS:
        candidate = add_scaled_variables(variables, direction, step)
        candidate = model.retract_variables(candidate)
        candidate_objective = model.objective(candidate)
        gauge_errors = model.gauge_errors(candidate)
        rows.append(
            {
                "step": float(step),
                "objective": float(candidate_objective),
                "objective_change": float(candidate_objective - objective),
                "max_abs_gauge_error": float(max_abs(gauge_errors)),
                "gauge_ok": bool(max_abs(gauge_errors) <= GAUGE_TOLERANCE),
                "downhill": bool(candidate_objective < objective),
            }
        )
    return rows


def analyze_start(
    model: RankOptimizationModel,
    name: str,
    variables: VariableDict,
) -> dict[str, Any]:
    objective = model.objective(variables)
    zero_retracted = model.retract_variables(variables)
    zero_retracted_objective = model.objective(zero_retracted)
    gradient = model.analytic_gradient(variables)
    plus_direction, plus_norm = signed_direction(model, variables, gradient, sign=1.0)
    minus_direction, minus_norm = signed_direction(model, variables, gradient, sign=-1.0)
    plus_derivative = variable_dot(gradient, plus_direction)
    minus_derivative = variable_dot(gradient, minus_direction)
    plus_trials = trial_rows(model, variables, plus_direction, objective)
    minus_trials = trial_rows(model, variables, minus_direction, objective)

    def best(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return min(rows, key=lambda item: item["objective"])

    return {
        "name": name,
        "objective": float(objective),
        "zero_retracted_objective": float(zero_retracted_objective),
        "zero_retraction_objective_change": float(zero_retracted_objective - objective),
        "gradient_norm": float(variable_norm(gradient)),
        "plus_gradient": {
            "projected_norm": float(plus_norm),
            "directional_derivative": float(plus_derivative),
            "best_trial": best(plus_trials),
            "trials": plus_trials,
        },
        "minus_gradient": {
            "projected_norm": float(minus_norm),
            "directional_derivative": float(minus_derivative),
            "best_trial": best(minus_trials),
            "trials": minus_trials,
        },
    }


def main() -> int:
    model = RankOptimizationModel(DATA_PATH)
    starts = [("baseline", copy_variables(model.variables))]
    if STATE_NPZ.exists():
        starts.append(("saved_final_state", load_state(STATE_NPZ)))

    results = {
        "description": "Constrained/retracted test of projected +gradient versus -gradient directions.",
        "steps": STEPS,
        "gauge_tolerance": GAUGE_TOLERANCE,
        "starts": [analyze_start(model, name, variables) for name, variables in starts],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("Gradient direction sign test")
    for start in results["starts"]:
        print(f"  {start['name']}: objective {start['objective']:.12e}")
        for label in ["plus_gradient", "minus_gradient"]:
            data = start[label]
            best = data["best_trial"]
            net_after_retraction = best["objective"] - start["zero_retracted_objective"]
            print(
                f"    {label:14s} deriv {data['directional_derivative']:.3e} "
                f"best step {best['step']:.1e} change {best['objective_change']:.3e} "
                f"net-after-retraction {net_after_retraction:.3e} "
                f"gauge {best['max_abs_gauge_error']:.1e}"
            )
    print(f"  wrote: {RESULTS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
