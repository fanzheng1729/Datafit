#!/usr/bin/env python3
"""Run row-band-scaled optimizer steps from a saved state.

This experiment starts from an existing optimizer state and runs the
all-variable row-band path, plus full-variable raw/vanilla controls.  The
row-band path uses direct scalar-gradient multiplier sweeps for cl/cw/rat and
measured row-band scaling for field blocks.

The objective itself is unchanged.  Every trial point is retracted/refit before
evaluation, exactly like the existing monitored optimizer.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np
from rank_optimizer_helpers import (
    GAUGE_TOLERANCE,
    INITIAL_STEP_SCALE,
    MIN_OBJECTIVE_DECREASE,
    STEP_MULTIPLIERS,
    candidate_summary_with_evaluation,
    choose_best_candidate,
    history_row,
    max_abs,
    negative_gradient,
    normalize_direction,
    output_paths,
    residual_rms_summary,
    save_state,
    write_history_csv,
)
from rank_optimization_model import (
    DATA_PATH,
    Evaluation,
    RankOptimizationModel,
    VariableDict,
    copy_variables,
    variable_dot,
    variable_norm,
)


DEFAULT_STATE = Path("from_begin_initial_state.npz")
DEFAULT_ROWBAND_DIAGNOSTIC = Path("all_variable_rowband_step_scaling_diagnostic_results.json")
DEFAULT_OUTPUT_PREFIX = "rowband_all_rank_optimization"
DEFAULT_CONSTRAINT_WEIGHT = 0.007
SCALAR_BLOCKS = ("cl", "cw", "rat")
DIRECT_FIELD_MULTIPLIERS = [
    10.0,
    3.0,
    1.0,
    0.3,
    0.1,
    0.03,
    0.01,
    0.003,
    0.001,
    0.0003,
    0.0001,
    0.00003,
    0.00001,
]
DIRECT_SCALAR_MULTIPLIERS = [
    0.0,
    1.0e-15,
    3.0e-15,
    1.0e-14,
    3.0e-14,
    1.0e-13,
    3.0e-13,
    1.0e-12,
    3.0e-12,
    1.0e-11,
    3.0e-11,
    1.0e-10,
    3.0e-10,
    1.0e-9,
    3.0e-9,
    1.0e-8,
    3.0e-8,
    1.0e-7,
    3.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
]


def load_state(path: Path) -> VariableDict:
    """Load a saved optimizer state produced by ``save_state``."""

    data = np.load(path, allow_pickle=True)
    variables: VariableDict = {}
    scalar_names = [str(name) for name in data["_scalar_names"]]
    scalar_values = [float(value) for value in data["_scalar_values"]]
    for key in data.files:
        if key.startswith("_"):
            continue
        variables[key] = data[key].copy()
    for key, value in zip(scalar_names, scalar_values):
        variables[key] = value
    return variables


def objective_components(model: RankOptimizationModel, residuals: dict[str, np.ndarray]) -> dict[str, float]:
    """Return the four weighted objective components separately."""

    return {
        "fomega": float(0.5 * np.mean((residuals["fomega"] / model.scales["fomega"]) ** 2)),
        "fzeta": float(0.5 * np.mean((residuals["fzeta"] / model.scales["fzeta"]) ** 2)),
        "divergence": float(
            0.5
            * model.constraint_weight
            * np.mean((residuals["divergence"] / model.scales["divergence"]) ** 2)
        ),
        "curl": float(
            0.5
            * model.constraint_weight
            * np.mean((residuals["curl"] / model.scales["curl"]) ** 2)
        ),
    }


def zero_like_variables(variables: VariableDict) -> VariableDict:
    """Return a zero variable dictionary with the same shape as ``variables``."""

    return {
        key: np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
        for key, value in variables.items()
    }


def target_blocks_from_diagnostic(rowband_diagnostic: dict[str, Any]) -> tuple[str, ...]:
    """Read target blocks from a diagnostic JSON.

    The diagnostic is authoritative because it records exactly which P/Q/s/scalar
    groups were probed and should therefore be moved.
    """

    target_blocks = rowband_diagnostic.get("target_blocks")
    if not target_blocks:
        raise ValueError("row-band diagnostic does not contain target_blocks")
    blocks = [str(block) for block in target_blocks]
    if "rat" not in blocks:
        blocks.append("rat")
    return tuple(blocks)


def selected_raw_direction(
    variables: VariableDict,
    gradient: VariableDict,
    target_blocks: tuple[str, ...],
) -> VariableDict:
    """Use only selected blocks of the unscaled negative gradient.

    This is a control mode for checking whether the benefit comes from the
    row-band safe-step scaling or merely from widening the variable set.
    """

    direction = zero_like_variables(variables)
    for block in target_blocks:
        value = gradient[block]
        direction[block] = -value.copy() if isinstance(value, np.ndarray) else -float(value)
    return direction


def set_group_scale(
    block_scale: np.ndarray | float,
    block_gradient: np.ndarray | float,
    rows: range | None,
    safe_step: float,
) -> np.ndarray | float:
    """Store ``safe_step / ||gradient group||`` for one block group.

    P/Q groups pass a row range, singular-value vectors pass all rows, and
    scalar rates pass ``rows=None``. The resulting scale is the diagonal
    row-band metric used for both the raw direction and gauge projection.
    """

    if isinstance(block_gradient, np.ndarray):
        assert isinstance(block_scale, np.ndarray)
        if rows is None:
            chunk = block_gradient
            norm = float(np.linalg.norm(chunk))
            if norm > 0.0:
                block_scale[...] = safe_step / norm
        elif block_gradient.ndim == 1:
            row_list = list(rows)
            chunk = block_gradient[row_list]
            norm = float(np.linalg.norm(chunk))
            if norm > 0.0:
                block_scale[row_list] = safe_step / norm
        else:
            row_list = list(rows)
            chunk = block_gradient[row_list, :]
            norm = float(np.linalg.norm(chunk))
            if norm > 0.0:
                block_scale[row_list, :] = safe_step / norm
        return block_scale

    gradient_value = float(block_gradient)
    norm = abs(gradient_value)
    if norm > 0.0:
        return safe_step / norm
    return block_scale


def rowband_preconditioner(
    variables: VariableDict,
    gradient: VariableDict,
    rowband_diagnostic: dict[str, Any],
    target_blocks: tuple[str, ...],
) -> VariableDict:
    """Build the measured positive row-band metric.

    The diagnostic JSON stores the largest accepted unit-norm step found for
    each P/Q coordinate row band and for each non-spatial block. This function
    stores the corresponding diagonal preconditioner scale for each group.
    """

    preconditioner = zero_like_variables(variables)
    for block in target_blocks:
        block_gradient = gradient[block]
        block_scale = preconditioner[block]
        if block not in rowband_diagnostic["block_results"]:
            if isinstance(block_gradient, np.ndarray):
                raise KeyError(f"row-band diagnostic does not contain block {block}")
            safe_step = fallback_scalar_safe_step(rowband_diagnostic)
            preconditioner[block] = set_group_scale(
                block_scale,
                block_gradient,
                None,
                safe_step,
            )
            continue
        for group in rowband_diagnostic["block_results"][block]["groups"]:
            safe_step = group["trial_summary"]["largest_accepted_step"]
            if safe_step is None:
                continue
            rows = (
                range(int(group["row_min"]), int(group["row_max"]) + 1)
                if isinstance(block_gradient, np.ndarray)
                else None
            )
            block_scale = set_group_scale(
                block_scale,
                block_gradient,
                rows,
                float(safe_step),
            )
        preconditioner[block] = block_scale
    return preconditioner


def apply_rowband_preconditioner(values: VariableDict, preconditioner: VariableDict) -> VariableDict:
    """Apply a variable-shaped diagonal row-band metric."""

    out = zero_like_variables(values)
    for key, value in values.items():
        scale = preconditioner[key]
        if isinstance(value, np.ndarray):
            assert isinstance(scale, np.ndarray)
            out[key] = scale * value
        else:
            out[key] = float(scale) * float(value)
    return out


def rowband_direction(gradient: VariableDict, preconditioner: VariableDict) -> VariableDict:
    """Return the row-band preconditioned negative gradient."""

    direction = apply_rowband_preconditioner(gradient, preconditioner)
    for key, value in direction.items():
        direction[key] = -value if isinstance(value, np.ndarray) else -float(value)
    return direction


def fallback_scalar_safe_step(rowband_diagnostic: dict[str, Any]) -> float:
    """Use the safest measured scalar-rate step for newly added scalars."""

    steps: list[float] = []
    for block in ("cl", "cw"):
        block_result = rowband_diagnostic.get("block_results", {}).get(block)
        if not block_result:
            continue
        for group in block_result["groups"]:
            safe_step = group["trial_summary"]["largest_accepted_step"]
            if safe_step is not None and safe_step > 0.0:
                steps.append(float(safe_step))
    return min(steps) if steps else 1.0e-7


def effective_direct_field_multipliers() -> list[float]:
    """Return the row-band field multiplier ladder for direct scalar sweeps."""

    return list(DIRECT_FIELD_MULTIPLIERS)


def effective_direct_scalar_multipliers() -> list[float]:
    """Return absolute scalar-gradient preconditioner multipliers."""

    return list(DIRECT_SCALAR_MULTIPLIERS)


def rowband_direct_base_preconditioner(
    variables: VariableDict,
    gradient: VariableDict,
    rowband_diagnostic: dict[str, Any],
    target_blocks: tuple[str, ...],
) -> VariableDict:
    """Build the row-band field metric with scalar movement disabled."""

    preconditioner = rowband_preconditioner(variables, gradient, rowband_diagnostic, target_blocks)
    for key in SCALAR_BLOCKS:
        if key in preconditioner:
            preconditioner[key] = 0.0
    return preconditioner


def rowband_direct_preconditioner(
    base_preconditioner: VariableDict,
    multipliers: dict[str, float],
) -> VariableDict:
    """Apply a field multiplier plus direct scalar-gradient multipliers."""

    out = copy_variables(base_preconditioner)
    field_multiplier = float(multipliers["field"])
    for key, value in out.items():
        if isinstance(value, np.ndarray):
            out[key] = field_multiplier * value
    for key in SCALAR_BLOCKS:
        out[key] = float(multipliers.get(key, 0.0))
    return out


def direct_multiplier_tuple(field: float, scalars: dict[str, float] | None = None) -> dict[str, float]:
    """Normalize direct row-band multiplier dictionaries for JSON/history."""

    scalars = scalars or {}
    return {
        "field": float(field),
        "cl": float(scalars.get("cl", 0.0)),
        "cw": float(scalars.get("cw", 0.0)),
        "rat": float(scalars.get("rat", 0.0)),
    }


def nearest_direct_multiplier_index(ladder: list[float], value: float) -> int:
    """Return the nearest direct multiplier index, allowing the zero endpoint."""

    if value <= 0.0:
        return 0
    positive = [
        (index, multiplier)
        for index, multiplier in enumerate(ladder)
        if multiplier > 0.0
    ]
    return min(
        positive,
        key=lambda item: abs(np.log(item[1]) - np.log(value)),
    )[0]


def raw_direction_for_mode(
    mode: str,
    variables: VariableDict,
    gradient: VariableDict,
    rowband_diagnostic: dict[str, Any] | None,
) -> tuple[VariableDict, VariableDict | None]:
    """Select the raw direction family requested by ``--mode``."""

    if mode == "vanilla":
        return negative_gradient(gradient), None
    if mode == "raw_all":
        if rowband_diagnostic is None:
            raise ValueError("raw_all mode requires a diagnostic JSON with target_blocks")
        return selected_raw_direction(variables, gradient, target_blocks_from_diagnostic(rowband_diagnostic)), None
    if mode == "rowband_all":
        if rowband_diagnostic is None:
            raise ValueError("rowband_all mode requires an all-variable diagnostic JSON")
        preconditioner = rowband_preconditioner(
            variables,
            gradient,
            rowband_diagnostic,
            target_blocks_from_diagnostic(rowband_diagnostic),
        )
        return rowband_direction(gradient, preconditioner), preconditioner
    raise ValueError(f"unknown mode: {mode}")


def projected_tangent_direction_for_mode(
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    raw_direction: VariableDict,
    preconditioner: VariableDict | None,
) -> VariableDict:
    """Project the direction into the fixed-gauge tangent space."""

    if preconditioner is None:
        return model.projected_tangent_direction(variables, raw_direction)
    return preconditioned_projected_tangent_direction(model, variables, gradient, preconditioner)


def preconditioned_projected_tangent_direction(
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    preconditioner: VariableDict,
) -> VariableDict:
    """Build ``d = -B(g + A lambda)`` with ``A.T d = 0``.

    ``B`` is the positive row-band metric from the diagnostic. Using it for
    both the gradient and gauge correction preserves the descent guarantee of
    the preconditioned direction after gauge projection.
    """

    gauges = model.gauge_gradients(variables)
    names = ["omega_x1_00", "theta_x1x1_00"]
    preconditioned_gradient = apply_rowband_preconditioner(gradient, preconditioner)
    projected: VariableDict = {
        key: -value if isinstance(value, np.ndarray) else -float(value)
        for key, value in preconditioned_gradient.items()
    }
    preconditioned_gauges = {
        name: apply_rowband_preconditioner(gauges[name], preconditioner)
        for name in names
    }
    gram = np.array(
        [
            [variable_dot(gauges[left], preconditioned_gauges[right]) for right in names]
            for left in names
        ],
        dtype=float,
    )
    rhs = np.array([variable_dot(gauges[name], projected) for name in names], dtype=float)
    if np.linalg.cond(gram) > 1.0e14:
        correction = np.linalg.pinv(gram) @ rhs
    else:
        correction = np.linalg.solve(gram, rhs)

    for coefficient, name in zip(correction, names):
        gauge_step = preconditioned_gauges[name]
        for key, value in projected.items():
            step_value = gauge_step[key]
            if isinstance(value, np.ndarray):
                assert isinstance(step_value, np.ndarray)
                value -= coefficient * step_value
            else:
                projected[key] = float(value) - float(coefficient) * float(step_value)
    return projected


def gauge_projection_name(preconditioner: VariableDict | None) -> str:
    """Return the projection label for result JSON."""

    return "preconditioned" if preconditioner is not None else "euclidean"


def effective_step_multipliers() -> list[float]:
    """Return the active multiplier ladder for the line search."""

    return list(STEP_MULTIPLIERS)


def line_search_steps(step_scale: float, multipliers: list[float]) -> list[float]:
    """Return trial step lengths for the current mode and trusted scale."""

    return [step_scale * multiplier for multiplier in multipliers]


def should_sweep_steps(iteration: int, *, step_sweep_initial_iterations: int, step_sweep_period: int) -> bool:
    """Return whether this iteration should search around the trusted step."""

    return (
        step_sweep_period == 1
        or iteration <= step_sweep_initial_iterations
        or iteration % step_sweep_period == 0
    )


def nearest_step_multiplier_index(ladder: list[float], value: float) -> int:
    """Return the closest multiplier index, using log distance."""

    return min(
        range(len(ladder)),
        key=lambda index: abs(np.log(ladder[index]) - np.log(value)),
    )


def neighbor_window_indices(center_index: int, ladder_length: int) -> list[int]:
    """Return old multiplier plus adjacent larger/smaller ladder entries."""

    lo = max(0, center_index - 1)
    hi = min(ladder_length - 1, center_index + 1)
    return list(range(lo, hi + 1))


def local_step_multiplier_window(
    *,
    last_step_multiplier: float | None,
    step_sweep_start_multiplier: float,
) -> list[float]:
    """Return the first local bracket for a neighbor sweep."""

    ladder = effective_step_multipliers()
    center = last_step_multiplier if last_step_multiplier is not None else step_sweep_start_multiplier
    center_index = nearest_step_multiplier_index(ladder, center)
    return [ladder[index] for index in neighbor_window_indices(center_index, len(ladder))]


def step_multipliers_for_iteration(
    *,
    iteration: int,
    step_sweep_initial_iterations: int,
    step_sweep_period: int,
    step_sweep_mode: str,
    step_sweep_start_multiplier: float,
    last_step_multiplier: float | None,
) -> tuple[list[float], str, bool]:
    """Choose a full bracket, neighbor bracket, or one trusted multiplier."""

    scheduled_sweep = should_sweep_steps(
        iteration,
        step_sweep_initial_iterations=step_sweep_initial_iterations,
        step_sweep_period=step_sweep_period,
    )
    if scheduled_sweep:
        if step_sweep_mode == "neighbor":
            return (
                local_step_multiplier_window(
                    last_step_multiplier=last_step_multiplier,
                    step_sweep_start_multiplier=step_sweep_start_multiplier,
                ),
                "neighbor_sweep",
                True,
            )
        return effective_step_multipliers(), "sweep", True
    if last_step_multiplier is not None:
        return [last_step_multiplier], "single", False
    return [1.0], "single_default", False


def candidate_with_components(
    model: RankOptimizationModel,
    variables: VariableDict,
    direction: VariableDict,
    step: float,
    base_objective: float,
    base_components: dict[str, float],
) -> tuple[dict[str, Any], VariableDict, Evaluation]:
    """Evaluate one candidate and attach per-equation component changes."""

    summary, candidate, evaluation = candidate_summary_with_evaluation(
        model,
        variables,
        direction,
        step,
        base_objective,
    )
    components = objective_components(model, evaluation.residuals)
    summary["objective_components"] = components
    summary["objective_component_changes"] = {
        key: components[key] - base_components[key]
        for key in components
    }
    return summary, candidate, evaluation


def direct_candidate_with_components(
    *,
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    base_preconditioner: VariableDict,
    multipliers: dict[str, float],
    base_objective: float,
    base_components: dict[str, float],
    iteration: int,
    step_search_mode: str,
) -> tuple[dict[str, Any], VariableDict, Evaluation] | None:
    """Evaluate one direct-scalar row-band multiplier tuple."""

    preconditioner = rowband_direct_preconditioner(base_preconditioner, multipliers)
    tangent_direction = preconditioned_projected_tangent_direction(
        model,
        variables,
        gradient,
        preconditioner,
    )
    direction, direction_norm = normalize_direction(tangent_direction)
    if direction_norm == 0.0:
        return None
    directional_derivative = variable_dot(gradient, direction)
    if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
        return None
    summary, candidate, evaluation = candidate_with_components(
        model,
        variables,
        direction,
        direction_norm,
        base_objective,
        base_components,
    )
    summary["iteration"] = iteration
    summary["step_multiplier"] = float(multipliers["field"])
    summary["step_multiplier_index"] = nearest_step_multiplier_index(
        effective_direct_field_multipliers(),
        float(multipliers["field"]),
    )
    summary["step_search_mode"] = step_search_mode
    summary["rowband_direct_multipliers"] = {
        key: float(multipliers[key])
        for key in ("field", *SCALAR_BLOCKS)
    }
    summary["field_step_multiplier"] = float(multipliers["field"])
    for key in SCALAR_BLOCKS:
        summary[f"{key}_direct_multiplier"] = float(multipliers[key])
    summary["directional_derivative"] = float(directional_derivative)
    return summary, candidate, evaluation


def evaluate_direct_tuple_candidates(
    *,
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    base_preconditioner: VariableDict,
    multiplier_tuples: list[dict[str, float]],
    base_objective: float,
    base_components: dict[str, float],
    iteration: int,
    step_search_mode: str,
) -> tuple[list[dict[str, Any]], list[VariableDict], list[Evaluation]]:
    """Evaluate direct row-band tuples, skipping non-descent degeneracies."""

    candidates: list[dict[str, Any]] = []
    candidate_variables: list[VariableDict] = []
    candidate_evaluations: list[Evaluation] = []
    seen: set[tuple[float, float, float, float]] = set()
    for multipliers in multiplier_tuples:
        key = tuple(float(multipliers[name]) for name in ("field", *SCALAR_BLOCKS))
        if key in seen:
            continue
        seen.add(key)
        result = direct_candidate_with_components(
            model=model,
            variables=variables,
            gradient=gradient,
            base_preconditioner=base_preconditioner,
            multipliers=multipliers,
            base_objective=base_objective,
            base_components=base_components,
            iteration=iteration,
            step_search_mode=step_search_mode,
        )
        if result is None:
            continue
        summary, candidate, evaluation = result
        candidates.append(summary)
        candidate_variables.append(candidate)
        candidate_evaluations.append(evaluation)
    return candidates, candidate_variables, candidate_evaluations


def direct_candidate_multipliers(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract a direct multiplier tuple from a candidate summary."""

    values = candidate["rowband_direct_multipliers"]
    return {
        key: float(values[key])
        for key in ("field", *SCALAR_BLOCKS)
    }


def rowband_direct_coordinate_candidates(
    *,
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    rowband_diagnostic: dict[str, Any],
    target_blocks: tuple[str, ...],
    field_multipliers: list[float],
    scalar_multipliers: list[float],
    base_objective: float,
    base_components: dict[str, float],
    iteration: int,
    step_search_mode: str,
) -> tuple[list[dict[str, Any]], list[VariableDict], list[Evaluation]]:
    """Field-only seed, then direct cl/cw/rat coordinate refinement."""

    base_preconditioner = rowband_direct_base_preconditioner(
        variables,
        gradient,
        rowband_diagnostic,
        target_blocks,
    )
    candidates: list[dict[str, Any]] = []
    candidate_variables: list[VariableDict] = []
    candidate_evaluations: list[Evaluation] = []

    seed_tuples = [
        direct_multiplier_tuple(field_multiplier)
        for field_multiplier in field_multipliers
    ]
    new_candidates, new_variables, new_evaluations = evaluate_direct_tuple_candidates(
        model=model,
        variables=variables,
        gradient=gradient,
        base_preconditioner=base_preconditioner,
        multiplier_tuples=seed_tuples,
        base_objective=base_objective,
        base_components=base_components,
        iteration=iteration,
        step_search_mode=step_search_mode,
    )
    candidates.extend(new_candidates)
    candidate_variables.extend(new_variables)
    candidate_evaluations.extend(new_evaluations)
    accepted_index, _ = choose_best_candidate(candidates, candidate_variables)
    if accepted_index is None:
        return candidates, candidate_variables, candidate_evaluations

    current_multipliers = direct_candidate_multipliers(candidates[accepted_index])
    for scalar_key in SCALAR_BLOCKS:
        trial_tuples = []
        for scalar_multiplier in scalar_multipliers:
            trial = dict(current_multipliers)
            trial[scalar_key] = float(scalar_multiplier)
            trial_tuples.append(trial)
        new_candidates, new_variables, new_evaluations = evaluate_direct_tuple_candidates(
            model=model,
            variables=variables,
            gradient=gradient,
            base_preconditioner=base_preconditioner,
            multiplier_tuples=trial_tuples,
            base_objective=base_objective,
            base_components=base_components,
            iteration=iteration,
            step_search_mode=f"{step_search_mode}_{scalar_key}",
        )
        candidates.extend(new_candidates)
        candidate_variables.extend(new_variables)
        candidate_evaluations.extend(new_evaluations)
        accepted_index, _ = choose_best_candidate(candidates, candidate_variables)
        if accepted_index is not None:
            current_multipliers = direct_candidate_multipliers(candidates[accepted_index])

    return candidates, candidate_variables, candidate_evaluations


def rowband_direct_single_candidate(
    *,
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    rowband_diagnostic: dict[str, Any],
    target_blocks: tuple[str, ...],
    multipliers: dict[str, float],
    base_objective: float,
    base_components: dict[str, float],
    iteration: int,
) -> tuple[list[dict[str, Any]], list[VariableDict], list[Evaluation]]:
    """Evaluate the trusted direct row-band multiplier tuple once."""

    base_preconditioner = rowband_direct_base_preconditioner(
        variables,
        gradient,
        rowband_diagnostic,
        target_blocks,
    )
    return evaluate_direct_tuple_candidates(
        model=model,
        variables=variables,
        gradient=gradient,
        base_preconditioner=base_preconditioner,
        multiplier_tuples=[multipliers],
        base_objective=base_objective,
        base_components=base_components,
        iteration=iteration,
        step_search_mode="direct_single",
    )


def component_history_fields(components: dict[str, float]) -> dict[str, float]:
    """Flatten component losses into CSV-friendly fields."""

    return {
        f"{name}_loss": float(value)
        for name, value in components.items()
    }


def evaluate_step_candidates(
    *,
    model: RankOptimizationModel,
    variables: VariableDict,
    direction: VariableDict,
    search_scale: float,
    step_multipliers: list[float],
    base_objective: float,
    base_components: dict[str, float],
    iteration: int,
    step_search_mode: str,
    step_multiplier_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[VariableDict], list[Evaluation]]:
    """Evaluate one line-search bracket or a single trusted step."""

    if step_multiplier_indices is None:
        ladder = effective_step_multipliers()
        step_multiplier_indices = [
            nearest_step_multiplier_index(ladder, multiplier)
            for multiplier in step_multipliers
        ]
    candidates: list[dict[str, Any]] = []
    candidate_variables: list[VariableDict] = []
    candidate_evaluations: list[Evaluation] = []
    for step, multiplier, multiplier_index in zip(
        line_search_steps(search_scale, step_multipliers),
        step_multipliers,
        step_multiplier_indices,
    ):
        summary, candidate, evaluation = candidate_with_components(
            model,
            variables,
            direction,
            step,
            base_objective,
            base_components,
        )
        summary["iteration"] = iteration
        summary["step_multiplier"] = float(multiplier)
        summary["step_multiplier_index"] = int(multiplier_index)
        summary["step_search_mode"] = step_search_mode
        candidates.append(summary)
        candidate_variables.append(candidate)
        candidate_evaluations.append(evaluation)
    return candidates, candidate_variables, candidate_evaluations


def evaluate_neighbor_step_candidates(
    *,
    model: RankOptimizationModel,
    variables: VariableDict,
    direction: VariableDict,
    search_scale: float,
    initial_multipliers: list[float],
    base_objective: float,
    base_components: dict[str, float],
    iteration: int,
) -> tuple[list[dict[str, Any]], list[VariableDict], list[Evaluation]]:
    """Walk the multiplier ladder until the best accepted candidate is interior."""

    ladder = effective_step_multipliers()
    current_indices = list(
        dict.fromkeys(
            nearest_step_multiplier_index(ladder, multiplier)
            for multiplier in initial_multipliers
        )
    )
    evaluated = [False] * len(ladder)
    candidates: list[dict[str, Any]] = []
    candidate_variables: list[VariableDict] = []
    candidate_evaluations: list[Evaluation] = []

    while True:
        new_indices = [index for index in current_indices if not evaluated[index]]
        if new_indices:
            new_candidates, new_variables, new_evaluations = evaluate_step_candidates(
                model=model,
                variables=variables,
                direction=direction,
                search_scale=search_scale,
                step_multipliers=[ladder[index] for index in new_indices],
                base_objective=base_objective,
                base_components=base_components,
                iteration=iteration,
                step_search_mode="neighbor_sweep",
                step_multiplier_indices=new_indices,
            )
            candidates.extend(new_candidates)
            candidate_variables.extend(new_variables)
            candidate_evaluations.extend(new_evaluations)
            for index in new_indices:
                evaluated[index] = True

        current_candidates = [
            candidate
            for candidate in candidates
            if int(candidate["step_multiplier_index"]) in current_indices
        ]
        accepted = [candidate for candidate in current_candidates if candidate["accepted"]]
        if not accepted:
            break

        best = min(accepted, key=lambda candidate: float(candidate["objective"]))
        best_index = int(best["step_multiplier_index"])
        if best_index == current_indices[0] and best_index > 0:
            current_indices = neighbor_window_indices(best_index, len(ladder))
        elif best_index == current_indices[-1] and best_index < len(ladder) - 1:
            current_indices = neighbor_window_indices(best_index, len(ladder))
        else:
            break

        if all(evaluated[index] for index in current_indices):
            break

    return candidates, candidate_variables, candidate_evaluations


def run_optimizer(
    *,
    mode: str,
    state_path: Path,
    rowband_diagnostic_path: Path | None,
    max_iterations: int,
    output_prefix: str,
    initial_step_scale: float,
    step_sweep_initial_iterations: int,
    step_sweep_period: int,
    step_sweep_mode: str,
    step_sweep_start_multiplier: float,
    recovery_step_sweep: bool,
    constraint_weight: float,
) -> dict[str, Any]:
    """Run a saved-state optimizer comparison.

    ``vanilla`` and ``raw_all`` modes are controls. ``rowband_all`` is the
    current recommended all-variable row-band setup.

    The objective and acceptance logic are deliberately unchanged across modes
    so the resulting JSON/CSV files differ only by the direction family and the
    chosen constraint weight.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if step_sweep_initial_iterations < 0:
        raise ValueError("step_sweep_initial_iterations must be nonnegative")
    if step_sweep_period < 1:
        raise ValueError("step_sweep_period must be positive")
    if step_sweep_mode not in ("full", "neighbor"):
        raise ValueError("step_sweep_mode must be 'full' or 'neighbor'")
    if step_sweep_start_multiplier <= 0.0 or not np.isfinite(step_sweep_start_multiplier):
        raise ValueError("step_sweep_start_multiplier must be positive and finite")

    model = RankOptimizationModel(DATA_PATH, constraint_weight=constraint_weight)
    variables = model.complete_variables(load_state(state_path))
    # Only row-band mode needs the diagnostic, but accepting an optional path
    # keeps the JSON output comparable across modes.
    rowband_diagnostic = None
    if rowband_diagnostic_path is not None and rowband_diagnostic_path.exists():
        rowband_diagnostic = json.loads(rowband_diagnostic_path.read_text(encoding="utf-8"))

    current_evaluation = model.evaluate(variables)
    initial_objective = model.objective_from_residuals(current_evaluation.residuals)
    initial_components = objective_components(model, current_evaluation.residuals)
    initial_residuals = residual_rms_summary(model, current_evaluation)
    initial_gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
    step_scale = initial_step_scale
    last_step_multiplier: float | None = None
    target_blocks = (
        target_blocks_from_diagnostic(rowband_diagnostic)
        if rowband_diagnostic is not None and mode in ("rowband_all", "raw_all")
        else ()
    )
    last_direct_multipliers: dict[str, float] | None = None

    history: list[dict[str, float | int | str]] = []
    candidate_history: list[dict[str, Any]] = []
    stop_reason = "reached maximum iterations"
    last_projection = "euclidean"
    paths = output_paths(output_prefix)

    for iteration in range(1, max_iterations + 1):
        base_objective = model.objective_from_residuals(current_evaluation.residuals)
        base_components = objective_components(model, current_evaluation.residuals)
        gradient = model.analytic_gradient(variables, evaluation=current_evaluation)
        gradient_norm = variable_norm(gradient)
        if mode == "rowband_all":
            if rowband_diagnostic is None:
                raise ValueError("rowband_all mode requires an all-variable diagnostic JSON")
            scheduled_sweep = should_sweep_steps(
                iteration,
                step_sweep_initial_iterations=step_sweep_initial_iterations,
                step_sweep_period=step_sweep_period,
            )
            search_scale = 1.0
            if scheduled_sweep or last_direct_multipliers is None:
                candidates, candidate_variables, candidate_evaluations = rowband_direct_coordinate_candidates(
                    model=model,
                    variables=variables,
                    gradient=gradient,
                    rowband_diagnostic=rowband_diagnostic,
                    target_blocks=target_blocks,
                    field_multipliers=effective_direct_field_multipliers(),
                    scalar_multipliers=effective_direct_scalar_multipliers(),
                    base_objective=base_objective,
                    base_components=base_components,
                    iteration=iteration,
                    step_search_mode="direct_sweep",
                )
            else:
                candidates, candidate_variables, candidate_evaluations = rowband_direct_single_candidate(
                    model=model,
                    variables=variables,
                    gradient=gradient,
                    rowband_diagnostic=rowband_diagnostic,
                    target_blocks=target_blocks,
                    multipliers=last_direct_multipliers,
                    base_objective=base_objective,
                    base_components=base_components,
                    iteration=iteration,
                )
        else:
            raw_direction, preconditioner = raw_direction_for_mode(mode, variables, gradient, rowband_diagnostic)
            tangent_direction = projected_tangent_direction_for_mode(
                model,
                variables,
                gradient,
                raw_direction,
                preconditioner,
            )
            direction, direction_norm_before_normalization = normalize_direction(tangent_direction)
            directional_derivative = variable_dot(gradient, direction)
            last_projection = gauge_projection_name(preconditioner)
            if (
                direction_norm_before_normalization == 0.0
                or not np.isfinite(directional_derivative)
                or directional_derivative >= 0.0
            ):
                stop_reason = "projected direction was not a descent direction"
                break

            search_scale = step_scale
            step_multipliers, step_search_mode, scheduled_sweep = step_multipliers_for_iteration(
                iteration=iteration,
                step_sweep_initial_iterations=step_sweep_initial_iterations,
                step_sweep_period=step_sweep_period,
                step_sweep_mode=step_sweep_mode,
                step_sweep_start_multiplier=step_sweep_start_multiplier,
                last_step_multiplier=last_step_multiplier,
            )
            if step_search_mode == "neighbor_sweep":
                candidates, candidate_variables, candidate_evaluations = evaluate_neighbor_step_candidates(
                    model=model,
                    variables=variables,
                    direction=direction,
                    search_scale=search_scale,
                    initial_multipliers=step_multipliers,
                    base_objective=base_objective,
                    base_components=base_components,
                    iteration=iteration,
                )
            else:
                candidates, candidate_variables, candidate_evaluations = evaluate_step_candidates(
                    model=model,
                    variables=variables,
                    direction=direction,
                    search_scale=search_scale,
                    step_multipliers=step_multipliers,
                    base_objective=base_objective,
                    base_components=base_components,
                    iteration=iteration,
                    step_search_mode=step_search_mode,
                )
        iteration_trial_count = len(candidates)
        candidate_history.extend(candidates)

        accepted_index, accepted_variables = choose_best_candidate(candidates, candidate_variables)
        if accepted_index is None and not scheduled_sweep and recovery_step_sweep:
            if mode == "rowband_all":
                recovery_candidates, recovery_variables, recovery_evaluations = rowband_direct_coordinate_candidates(
                    model=model,
                    variables=variables,
                    gradient=gradient,
                    rowband_diagnostic=rowband_diagnostic,
                    target_blocks=target_blocks,
                    field_multipliers=effective_direct_field_multipliers(),
                    scalar_multipliers=effective_direct_scalar_multipliers(),
                    base_objective=base_objective,
                    base_components=base_components,
                    iteration=iteration,
                    step_search_mode="direct_recovery_sweep",
                )
            else:
                recovery_candidates, recovery_variables, recovery_evaluations = evaluate_step_candidates(
                    model=model,
                    variables=variables,
                    direction=direction,
                    search_scale=search_scale,
                    step_multipliers=effective_step_multipliers(),
                    base_objective=base_objective,
                    base_components=base_components,
                    iteration=iteration,
                    step_search_mode="recovery_sweep",
                )
            iteration_trial_count += len(recovery_candidates)
            candidate_history.extend(recovery_candidates)
            candidates = recovery_candidates
            candidate_variables = recovery_variables
            candidate_evaluations = recovery_evaluations
            accepted_index, accepted_variables = choose_best_candidate(candidates, candidate_variables)
        if accepted_index is None or accepted_variables is None:
            stop_reason = "no line-search candidate reduced the objective while preserving gauges"
            break

        best = candidates[accepted_index]
        variables = accepted_variables
        current_evaluation = candidate_evaluations[accepted_index]
        # Preserve the best accepted scalar as the next trusted multiplier.
        # Row-band mode resets the physical scale from the natural direction
        # norm each iteration, while still reusing this multiplier.
        step_scale = float(best["step"])
        last_step_multiplier = float(best["step_multiplier"])
        if mode == "rowband_all":
            last_direct_multipliers = direct_candidate_multipliers(best)
            direction_norm_before_normalization = float(best["step"])
            directional_derivative = float(best["directional_derivative"])
            last_projection = "preconditioned"
        residuals = residual_rms_summary(model, current_evaluation)
        gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
        components = best["objective_components"]
        row = history_row(
            iteration=iteration,
            objective=float(best["objective"]),
            objective_change=float(best["objective_change"]),
            accepted_step=float(best["step"]),
            gradient_norm=gradient_norm,
            projected_gradient_norm=direction_norm_before_normalization,
            directional_derivative=directional_derivative,
            gauge_errors=gauge_errors,
            residuals=residuals,
        )
        row["mode"] = mode
        row["search_step_scale"] = float(search_scale)
        row["step_search_mode"] = str(best["step_search_mode"])
        row["accepted_step_multiplier"] = float(best["step_multiplier"])
        if mode == "rowband_all":
            row["field_step_multiplier"] = float(best["field_step_multiplier"])
            for key in SCALAR_BLOCKS:
                row[f"{key}_direct_multiplier"] = float(best[f"{key}_direct_multiplier"])
        row["line_search_trial_count"] = int(iteration_trial_count)
        row.update(component_history_fields(components))
        history.append(row)
        print(
            f"  iter {iteration:02d}/{max_iterations}: "
            f"J={best['objective']:.12e} "
            f"dJ={best['objective_change']:.3e} "
            f"step={best['step']:.3e} "
            f"mult={best['step_multiplier']:.3g} "
            f"search={best['step_search_mode']} "
            f"trials={iteration_trial_count} "
            f"|d_raw|={direction_norm_before_normalization:.3e} "
            f"div_loss={components['divergence']:.3e} "
            f"curl_loss={components['curl']:.3e} "
            f"gauge={max_abs(gauge_errors):.3e}",
            flush=True,
        )

        if abs(float(best["objective_change"])) < MIN_OBJECTIVE_DECREASE:
            stop_reason = "accepted decrease fell below the minimum improvement"
            break

    final_objective = model.objective_from_residuals(current_evaluation.residuals)
    final_components = objective_components(model, current_evaluation.residuals)
    final_residuals = residual_rms_summary(model, current_evaluation)
    final_gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
    accepted_count = len(history)
    if accepted_count > 0:
        # Save the accepted endpoint so users can inspect or continue the exact
        # state corresponding to the JSON/CSV result.
        save_state(paths["state"], variables)

    return {
        "description": "Saved-state rank-factor optimizer comparison with all-variable row-band direction scaling.",
        "mode": mode,
        "constraint_weight": constraint_weight,
        "state_path": str(state_path),
        "rowband_diagnostic_path": str(rowband_diagnostic_path) if rowband_diagnostic_path is not None else None,
        "accepted_steps": accepted_count,
        "state_file": paths["state"].name if accepted_count > 0 else None,
        "history_file": paths["history"].name,
        "stop_reason": stop_reason,
        "objective_before": float(initial_objective),
        "objective_after": float(final_objective),
        "objective_change": float(final_objective - initial_objective),
        "relative_objective_change": float((final_objective - initial_objective) / initial_objective),
        "objective_components_before": initial_components,
        "objective_components_after": final_components,
        "initial_residual_rms": initial_residuals,
        "final_residual_rms": final_residuals,
        "gauge_targets": model.gauge_targets,
        "gauge_errors_before": initial_gauge_errors,
        "gauge_errors_after": final_gauge_errors,
        "gauge_projection": last_projection,
        "initial_step_scale": initial_step_scale,
        "line_search_step_multipliers": effective_step_multipliers(),
        "rowband_scalar_update": "direct_gradient_coordinate_sweep" if mode == "rowband_all" else "global_line_search",
        "direct_field_multipliers": effective_direct_field_multipliers() if mode == "rowband_all" else [],
        "direct_scalar_multipliers": effective_direct_scalar_multipliers() if mode == "rowband_all" else [],
        "last_direct_multipliers": last_direct_multipliers if last_direct_multipliers is not None else {},
        "step_sweep_initial_iterations": step_sweep_initial_iterations,
        "step_sweep_period": step_sweep_period,
        "step_sweep_mode": step_sweep_mode,
        "step_sweep_start_multiplier": step_sweep_start_multiplier,
        "recovery_step_sweep": recovery_step_sweep,
        "max_iterations": max_iterations,
        "min_objective_decrease": MIN_OBJECTIVE_DECREASE,
        "gauge_tolerance": GAUGE_TOLERANCE,
        "target_blocks": target_blocks,
        "ranks": {core.name: int(core.rank) for core in model.cores},
        "history": history,
        "candidate_history": candidate_history,
    }


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Parse saved-state comparison options."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("vanilla", "raw_all", "rowband_all"),
        default="rowband_all",
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--rowband-diagnostic", type=Path, default=DEFAULT_ROWBAND_DIAGNOSTIC)
    parser.add_argument("-n", "--max-iterations", type=int, default=30)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--initial-step-scale", type=float, default=INITIAL_STEP_SCALE)
    parser.add_argument("--constraint-weight", type=float, default=DEFAULT_CONSTRAINT_WEIGHT)
    parser.add_argument(
        "--step-sweep-initial-iterations",
        type=int,
        default=5,
        help="number of initial iterations that search around the step multiplier",
    )
    parser.add_argument(
        "--step-sweep-period",
        type=int,
        default=5,
        help="after warmup, search around the step multiplier every N iterations",
    )
    parser.add_argument(
        "--step-sweep-mode",
        choices=("full", "neighbor"),
        default="neighbor",
        help="use the full ladder or the local neighbor-walk ladder on scheduled sweeps",
    )
    parser.add_argument(
        "--step-sweep-start-multiplier",
        type=float,
        default=1.0,
        help="starting multiplier for the first neighbor sweep before any accepted step exists",
    )
    parser.add_argument(
        "--no-recovery-step-sweep",
        action="store_true",
        help="stop instead of trying a full ladder sweep when an unscheduled single step fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = parse_args(argv)
    np.set_printoptions(precision=10, suppress=False)
    output = run_optimizer(
        mode=args.mode,
        state_path=args.state,
        rowband_diagnostic_path=args.rowband_diagnostic,
        max_iterations=args.max_iterations,
        output_prefix=args.output_prefix,
        initial_step_scale=args.initial_step_scale,
        step_sweep_initial_iterations=args.step_sweep_initial_iterations,
        step_sweep_period=args.step_sweep_period,
        step_sweep_mode=args.step_sweep_mode,
        step_sweep_start_multiplier=args.step_sweep_start_multiplier,
        recovery_step_sweep=not args.no_recovery_step_sweep,
        constraint_weight=args.constraint_weight,
    )
    paths = output_paths(args.output_prefix)
    write_history_csv(paths["history"], output["history"])
    paths["results"].write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"  accepted steps:            {output['accepted_steps']}")
    print(f"  stop reason:               {output['stop_reason']}")
    print(f"  objective before:          {output['objective_before']:.12e}")
    print(f"  objective after:           {output['objective_after']:.12e}")
    print(f"  objective change:          {output['objective_change']:.12e}")
    print(f"  wrote history:             {paths['history'].name}")
    if output["state_file"] is not None:
        print(f"  wrote state:               {output['state_file']}")
    print(f"  wrote results:             {paths['results'].name}")
    return 0 if output["accepted_steps"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
