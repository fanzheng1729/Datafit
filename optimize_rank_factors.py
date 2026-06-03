#!/usr/bin/env python3
"""Monitored explicit-S rank-factor gradient optimizer.

The optimizer uses the shared ``RankOptimizationModel``:

1. evaluate objective and analytic gradient,
2. project the descent direction onto the two fixed-gauge tangent constraints,
3. normalize that direction in the current Euclidean coefficient metric,
4. line-search over very small steps,
5. retract/refit every trial point back to the normalized explicit-S chart.

The metric is still primitive, so this is a conservative gradient optimizer
rather than a production Newton or quasi-Newton method.  The history is written
with enough diagnostics to show whether the process is genuinely descending and
whether the fixed gauges remain pinned.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np

from rank_optimization_model import (
    DATA_PATH,
    Evaluation,
    RankOptimizationModel,
    VariableDict,
    add_scaled_variables,
    copy_variables,
    variable_dot,
    variable_norm,
)


DEFAULT_OUTPUT_PREFIX = "rank_optimization"
DEFAULT_MAX_ITERATIONS = 20

# A candidate is accepted only when it beats the current objective by this much
# and preserves the two fixed gauges.  The tolerance is deliberately above
# roundoff because many trial steps are extremely small.
MIN_OBJECTIVE_DECREASE = 1.0e-10
GAUGE_TOLERANCE = 1.0e-8

# Steps are tiny because the coefficient-space gradient is very large in the
# velocity derivative blocks.  The line search reuses the previous accepted
# scale and probes a small bracket around it.
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


def line_search_steps(step_scale: float) -> list[float]:
    """Return the trial step lengths around the current trusted scale."""

    return [step_scale * multiplier for multiplier in STEP_MULTIPLIERS]


def candidate_summary(
    model: RankOptimizationModel,
    base_variables: VariableDict,
    direction: VariableDict,
    step: float,
    base_objective: float,
) -> tuple[dict[str, Any], VariableDict]:
    """Evaluate a trial step and return only the JSON-safe summary plus state."""

    summary, candidate, _ = candidate_summary_with_evaluation(
        model, base_variables, direction, step, base_objective
    )
    return summary, candidate


def candidate_summary_with_evaluation(
    model: RankOptimizationModel,
    base_variables: VariableDict,
    direction: VariableDict,
    step: float,
    base_objective: float,
) -> tuple[dict[str, Any], VariableDict, Evaluation]:
    """Evaluate, retract, and classify one line-search candidate.

    Every candidate is retracted before evaluation so the line search measures
    objective changes on the same explicit-S chart used after accepting a step.
    """

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
    """Persist the mixed variable dictionary to a compressed NPZ file.

    Arrays are stored under their variable names.  Scalars are packed into two
    metadata arrays so ``load_state`` helpers can reconstruct the same
    dictionary without guessing which entries were scalar.
    """

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
    """Resolve result/history/state file names next to this script."""

    stem = Path(prefix).name
    return {
        "results": Path(__file__).with_name(f"{stem}_results.json"),
        "history": Path(__file__).with_name(f"{stem}_history.csv"),
        "state": Path(__file__).with_name(f"{stem}_state.npz"),
    }


def run_optimizer(
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Run the monitored vanilla projected-gradient optimizer."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    # Start from the rank factors fitted from data.mat.  More advanced scripts
    # can resume from saved states, but this baseline deliberately starts from
    # the canonical MAT-file fit.
    model = RankOptimizationModel(DATA_PATH)
    variables = copy_variables(model.variables)
    current_evaluation = model.evaluate(variables)
    initial_objective = model.objective_from_residuals(current_evaluation.residuals)
    initial_residuals = residual_rms_summary(model, current_evaluation)
    initial_gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
    step_scale = INITIAL_STEP_SCALE
    history: list[dict[str, float | int]] = []
    candidate_history: list[dict[str, Any]] = []
    stop_reason = "reached maximum iterations"

    paths = output_paths(output_prefix)

    for iteration in range(1, max_iterations + 1):
        base_objective = model.objective_from_residuals(current_evaluation.residuals)

        # Project before normalizing so the step length has the same meaning
        # after removing the forbidden gauge directions.
        gradient = model.analytic_gradient(variables, evaluation=current_evaluation)
        raw_direction = negative_gradient(gradient)
        tangent_direction = model.projected_tangent_direction(variables, raw_direction)
        direction, projected_gradient_norm = normalize_direction(tangent_direction)
        gradient_norm = variable_norm(gradient)
        directional_derivative = variable_dot(gradient, direction)
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            stop_reason = "projected direction was not a descent direction"
            break

        # Probe a small multiplicative bracket around the previous accepted
        # scale.  The best passing candidate is chosen, not merely the first
        # decrease, because overshoot can be highly nonmonotone here.
        candidates: list[dict[str, Any]] = []
        candidate_variables: list[VariableDict] = []
        candidate_evaluations: list[Evaluation] = []
        for step in line_search_steps(step_scale):
            summary, candidate, evaluation = candidate_summary_with_evaluation(
                model, variables, direction, step, base_objective
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
        # The next iteration trusts the scale that just worked.  This gives the
        # "variable step size" behavior while keeping the search very simple.
        step_scale = best["step"]
        residuals = residual_rms_summary(model, current_evaluation)
        gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
        history.append(
            history_row(
                iteration=iteration,
                objective=float(best["objective"]),
                objective_change=float(best["objective_change"]),
                accepted_step=float(best["step"]),
                gradient_norm=gradient_norm,
                projected_gradient_norm=projected_gradient_norm,
                directional_derivative=directional_derivative,
                gauge_errors=gauge_errors,
                residuals=residuals,
            )
        )
        print(
            f"  iter {iteration:02d}/{max_iterations}: "
            f"J={best['objective']:.12e} "
            f"dJ={best['objective_change']:.3e} "
            f"step={best['step']:.3e} "
            f"|g_T|={projected_gradient_norm:.3e} "
            f"curl={residuals['curl']:.3e} "
            f"gauge={max_abs(gauge_errors):.3e}",
            flush=True,
        )

        if abs(float(best["objective_change"])) < MIN_OBJECTIVE_DECREASE:
            stop_reason = "accepted decrease fell below the minimum improvement"
            break

    final_objective = model.objective_from_residuals(current_evaluation.residuals)
    final_residuals = residual_rms_summary(model, current_evaluation)
    final_gauge_errors = model.gauge_errors_from_fields(current_evaluation.fields)
    accepted_count = len(history)
    if accepted_count > 0:
        # The final state is optional because failed runs still produce a JSON
        # diagnostic and empty history file, but should not overwrite a useful
        # state snapshot.
        save_state(paths["state"], variables)

    return {
        "description": "Monitored explicit-S rank-factor gradient optimization.",
        "accepted_steps": accepted_count,
        "state_file": paths["state"].name if accepted_count > 0 else None,
        "history_file": paths["history"].name,
        "stop_reason": stop_reason,
        "objective_before": float(initial_objective),
        "objective_after": float(final_objective),
        "objective_change": float(final_objective - initial_objective),
        "relative_objective_change": float((final_objective - initial_objective) / initial_objective),
        "initial_residual_rms": initial_residuals,
        "final_residual_rms": final_residuals,
        "gauge_targets": model.gauge_targets,
        "gauge_errors_before": initial_gauge_errors,
        "gauge_errors_after": final_gauge_errors,
        "line_search_step_multipliers": STEP_MULTIPLIERS,
        "initial_step_scale": INITIAL_STEP_SCALE,
        "max_iterations": max_iterations,
        "min_objective_decrease": MIN_OBJECTIVE_DECREASE,
        "gauge_tolerance": GAUGE_TOLERANCE,
        "ranks": {core.name: int(core.rank) for core in model.cores},
        "history": history,
        "candidate_history": candidate_history,
    }


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Parse command-line options without doing any numerical work."""

    parser = ArgumentParser(
        description="Run the monitored explicit-S rank-factor gradient optimizer."
    )
    parser.add_argument(
        "-n",
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"maximum optimizer iterations to attempt (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "prefix for saved JSON/CSV/NPZ files "
            f"(default: {DEFAULT_OUTPUT_PREFIX})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = parse_args(argv)
    np.set_printoptions(precision=10, suppress=False)
    print("Monitored explicit-S rank-factor gradient optimization")
    print(f"  max iterations:           {args.max_iterations}")
    print(f"  output prefix:            {Path(args.output_prefix).name}")
    output = run_optimizer(
        max_iterations=args.max_iterations,
        output_prefix=args.output_prefix,
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
