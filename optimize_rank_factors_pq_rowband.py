#!/usr/bin/env python3
"""Compare vanilla and row-band-scaled P/Q optimizer steps from a saved state.

This experiment starts from an existing optimizer state and runs one of:

* ``vanilla``: the full negative analytic gradient, as in the plain optimizer.
* ``raw_pq``: only the raw negative gradient on u1/u2 P/Q blocks.
* ``rowband_pq``: only u1/u2 P/Q blocks, with each coordinate row band scaled
  by the largest accepted unit step measured by
  ``diagnose_pq_support_step_scaling.py``.

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
from optimize_rank_factors import (
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
    CONSTRAINT_WEIGHT,
    DATA_PATH,
    Evaluation,
    RankOptimizationModel,
    VariableDict,
    copy_variables,
    variable_dot,
    variable_norm,
)


DEFAULT_STATE = Path("compare_curl_trust_rank_optimization_state.npz")
DEFAULT_PQ_DIAGNOSTIC = Path("pq_support_step_scaling_diagnostic_results.json")
DEFAULT_OUTPUT_PREFIX = "pq_rowband_rank_optimization"
TARGET_BLOCKS = ("u1_P", "u1_Q", "u2_P", "u2_Q")
EXTRA_SHRINKS = [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]


def load_state(path: Path) -> VariableDict:
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
    return {
        key: np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
        for key, value in variables.items()
    }


def raw_pq_direction(variables: VariableDict, gradient: VariableDict) -> VariableDict:
    direction = zero_like_variables(variables)
    for block in TARGET_BLOCKS:
        value = gradient[block]
        assert isinstance(value, np.ndarray)
        direction[block] = -value
    return direction


def rowband_pq_direction(
    variables: VariableDict,
    gradient: VariableDict,
    pq_diagnostic: dict[str, Any],
) -> VariableDict:
    direction = zero_like_variables(variables)
    for block in TARGET_BLOCKS:
        block_gradient = gradient[block]
        assert isinstance(block_gradient, np.ndarray)
        block_direction = np.zeros_like(block_gradient)
        for group in pq_diagnostic["block_results"][block]["groups"]:
            safe_step = group["trial_summary"]["largest_accepted_step"]
            if safe_step is None:
                continue
            rows = list(range(group["row_min"], group["row_max"] + 1))
            chunk = block_gradient[rows, :]
            norm = float(np.linalg.norm(chunk))
            if norm > 0.0:
                block_direction[rows, :] = -chunk * (safe_step / norm)
        direction[block] = block_direction
    return direction


def raw_direction_for_mode(
    mode: str,
    variables: VariableDict,
    gradient: VariableDict,
    pq_diagnostic: dict[str, Any] | None,
) -> VariableDict:
    if mode == "vanilla":
        return negative_gradient(gradient)
    if mode == "raw_pq":
        return raw_pq_direction(variables, gradient)
    if mode == "rowband_pq":
        if pq_diagnostic is None:
            raise ValueError("rowband_pq mode requires a P/Q diagnostic JSON")
        return rowband_pq_direction(variables, gradient, pq_diagnostic)
    raise ValueError(f"unknown mode: {mode}")


def line_search_steps(step_scale: float, *, include_extra_shrinks: bool) -> list[float]:
    multipliers = list(STEP_MULTIPLIERS)
    if include_extra_shrinks:
        multipliers.extend(EXTRA_SHRINKS)
    return [step_scale * multiplier for multiplier in multipliers]


def candidate_with_components(
    model: RankOptimizationModel,
    variables: VariableDict,
    direction: VariableDict,
    step: float,
    base_objective: float,
    base_components: dict[str, float],
) -> tuple[dict[str, Any], VariableDict, Evaluation]:
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


def component_history_fields(components: dict[str, float]) -> dict[str, float]:
    return {
        f"{name}_loss": float(value)
        for name, value in components.items()
    }


def run_optimizer(
    *,
    mode: str,
    state_path: Path,
    pq_diagnostic_path: Path | None,
    max_iterations: int,
    output_prefix: str,
    initial_step_scale: float,
    rowband_use_natural_step: bool,
    include_extra_shrinks: bool,
    constraint_weight: float,
) -> dict[str, Any]:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    model = RankOptimizationModel(DATA_PATH, constraint_weight=constraint_weight)
    variables = load_state(state_path)
    pq_diagnostic = None
    if pq_diagnostic_path is not None and pq_diagnostic_path.exists():
        pq_diagnostic = json.loads(pq_diagnostic_path.read_text(encoding="utf-8"))

    current_evaluation = model.evaluate(variables)
    initial_objective = model.objective_from_residuals(current_evaluation.residuals)
    initial_components = objective_components(model, current_evaluation.residuals)
    initial_residuals = residual_rms_summary(model, current_evaluation)
    initial_gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
    step_scale = initial_step_scale

    history: list[dict[str, float | int | str]] = []
    candidate_history: list[dict[str, Any]] = []
    stop_reason = "reached maximum iterations"
    paths = output_paths(output_prefix)

    for iteration in range(1, max_iterations + 1):
        base_objective = model.objective_from_residuals(current_evaluation.residuals)
        base_components = objective_components(model, current_evaluation.residuals)
        gradient = model.analytic_gradient(variables, evaluation=current_evaluation)
        raw_direction = raw_direction_for_mode(mode, variables, gradient, pq_diagnostic)
        tangent_direction = model.projected_tangent_direction(variables, raw_direction)
        direction, direction_norm_before_normalization = normalize_direction(tangent_direction)
        gradient_norm = variable_norm(gradient)
        directional_derivative = variable_dot(gradient, direction)
        if (
            direction_norm_before_normalization == 0.0
            or not np.isfinite(directional_derivative)
            or directional_derivative >= 0.0
        ):
            stop_reason = "projected direction was not a descent direction"
            break

        search_scale = step_scale
        if mode == "rowband_pq" and rowband_use_natural_step:
            search_scale = direction_norm_before_normalization

        candidates: list[dict[str, Any]] = []
        candidate_variables: list[VariableDict] = []
        candidate_evaluations: list[Evaluation] = []
        for step in line_search_steps(search_scale, include_extra_shrinks=include_extra_shrinks):
            summary, candidate, evaluation = candidate_with_components(
                model,
                variables,
                direction,
                step,
                base_objective,
                base_components,
            )
            summary["iteration"] = iteration
            candidates.append(summary)
            candidate_variables.append(candidate)
            candidate_evaluations.append(evaluation)
        candidate_history.extend(candidates)

        accepted_index, accepted_variables = choose_best_candidate(candidates, candidate_variables)
        if accepted_index is None or accepted_variables is None:
            stop_reason = "no line-search candidate reduced the objective while preserving gauges"
            break

        best = candidates[accepted_index]
        variables = accepted_variables
        current_evaluation = candidate_evaluations[accepted_index]
        step_scale = float(best["step"])
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
        row.update(component_history_fields(components))
        history.append(row)
        print(
            f"  iter {iteration:02d}/{max_iterations}: "
            f"J={best['objective']:.12e} "
            f"dJ={best['objective_change']:.3e} "
            f"step={best['step']:.3e} "
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
        save_state(paths["state"], variables)

    return {
        "description": "Saved-state rank-factor optimizer comparison with optional P/Q row-band direction scaling.",
        "mode": mode,
        "constraint_weight": constraint_weight,
        "state_path": str(state_path),
        "pq_diagnostic_path": str(pq_diagnostic_path) if pq_diagnostic_path is not None else None,
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
        "initial_step_scale": initial_step_scale,
        "rowband_use_natural_step": rowband_use_natural_step,
        "include_extra_shrinks": include_extra_shrinks,
        "line_search_step_multipliers": list(STEP_MULTIPLIERS)
        + (EXTRA_SHRINKS if include_extra_shrinks else []),
        "max_iterations": max_iterations,
        "min_objective_decrease": MIN_OBJECTIVE_DECREASE,
        "gauge_tolerance": GAUGE_TOLERANCE,
        "target_blocks": TARGET_BLOCKS,
        "ranks": {core.name: int(core.rank) for core in model.cores},
        "history": history,
        "candidate_history": candidate_history,
    }


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("vanilla", "raw_pq", "rowband_pq"),
        default="rowband_pq",
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--pq-diagnostic", type=Path, default=DEFAULT_PQ_DIAGNOSTIC)
    parser.add_argument("-n", "--max-iterations", type=int, default=10)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--initial-step-scale", type=float, default=INITIAL_STEP_SCALE)
    parser.add_argument("--constraint-weight", type=float, default=CONSTRAINT_WEIGHT)
    parser.add_argument(
        "--fixed-rowband-step-scale",
        action="store_true",
        help="use --initial-step-scale for rowband mode instead of the raw rowband direction norm",
    )
    parser.add_argument(
        "--no-extra-shrinks",
        action="store_true",
        help="only use the original seven vanilla line-search multipliers",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.set_printoptions(precision=10, suppress=False)
    output = run_optimizer(
        mode=args.mode,
        state_path=args.state,
        pq_diagnostic_path=args.pq_diagnostic,
        max_iterations=args.max_iterations,
        output_prefix=args.output_prefix,
        initial_step_scale=args.initial_step_scale,
        rowband_use_natural_step=not args.fixed_rowband_step_scale,
        include_extra_shrinks=not args.no_extra_shrinks,
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
