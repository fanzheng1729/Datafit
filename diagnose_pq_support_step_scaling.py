#!/usr/bin/env python3
"""Diagnose row-band and block step scales for rank-factor variables.

The P coefficients multiply the left B-spline basis and are localized along
``x1``.  The Q coefficients multiply the right B-spline basis and are localized
along ``x2``.  This script maps coefficient rows to their grid support, groups
rows by coordinate range, probes block-row-group step sizes, and fits a simple
power law

    safe_step ~= C * (1 + axis_coordinate^2)^alpha

for each spatial P/Q block. It also probes the non-spatial ``s``, ``cl``,
``cw``, and ``rat`` variables as single blocks so an optimizer can move every
variable while still using measured safe scales.
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
    MIN_OBJECTIVE_DECREASE,
    candidate_summary_with_evaluation,
    max_abs,
    normalize_direction,
    residual_rms_summary,
)
from rank_optimization_model import (
    CONSTRAINT_WEIGHT,
    DATA_PATH,
    RankCore,
    RankOptimizationModel,
    VariableDict,
    variable_dot,
)


DEFAULT_STATE = Path("from_begin_initial_state.npz")
DEFAULT_OUTPUT = Path("all_variable_rowband_step_scaling_diagnostic_results.json")
DEFAULT_STEPS = [
    1.0e-5,
    3.0e-6,
    1.0e-6,
    3.0e-7,
    1.0e-7,
    3.0e-8,
    1.0e-8,
    3.0e-9,
    1.0e-9,
    3.0e-10,
    1.0e-10,
    3.0e-11,
    1.0e-11,
    3.0e-12,
    1.0e-12,
    3.0e-13,
    1.0e-13,
    3.0e-14,
    1.0e-14,
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
]
DEFAULT_AXIS_EDGES = [0.02, 0.1, 1.0, 10.0, 100.0, 1.0e4, 1.0e8, 1.0e12]


def parse_steps(raw: str | None) -> list[float]:
    if raw is None:
        return DEFAULT_STEPS
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_edges(raw: str | None) -> list[float]:
    if raw is None:
        return DEFAULT_AXIS_EDGES
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def zero_like_variables(variables: VariableDict) -> VariableDict:
    """Return a variable dictionary whose arrays/scalars are all zero.

    Group probes isolate one block at a time.  Starting from a zero-shaped
    dictionary makes it explicit that all untouched blocks are held fixed before
    the usual gauge projection is applied.
    """

    out: VariableDict = {}
    for key, value in variables.items():
        out[key] = np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
    return out


def load_state(path: Path) -> VariableDict:
    """Load an optimizer NPZ state saved by ``rank_optimizer_helpers.save_state``.

    The repository keeps the useful starting state in LFS.  Reimplementing this
    small loader here keeps the diagnostic self-contained rather than depending
    on the local L-BFGS experiment script.
    """

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
    """Return the four terms that sum to the normalized objective.

    The all-variable sweep cares about tradeoffs, especially the profile losses
    versus divergence/curl.  Recording component losses in every probe makes the
    diagnostic useful even when a candidate is rejected by the aggregate loss.
    """

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


def variable_block_summary(values: VariableDict) -> dict[str, Any]:
    """Summarize array/scalar block gradient sizes for JSON output."""

    rows: list[dict[str, Any]] = []
    total_sq = 0.0
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            norm = float(np.linalg.norm(value))
            max_abs = float(np.max(np.abs(value))) if value.size else 0.0
            shape = list(value.shape)
        else:
            norm = abs(float(value))
            max_abs = abs(float(value))
            shape = []
        total_sq += norm**2
        rows.append({"block": key, "shape": shape, "norm": norm, "max_abs": max_abs})
    total = float(np.sqrt(total_sq))
    for row in rows:
        row["norm_fraction"] = float(row["norm"] / max(total, np.finfo(float).tiny))
        row["l2_mass_fraction"] = float(row["norm"] ** 2 / max(total_sq, np.finfo(float).tiny))
    rows.sort(key=lambda item: item["norm"], reverse=True)
    return {"total_norm": total, "blocks": rows}


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a step ladder into the safe-step quantities used downstream."""

    decreasing = [row for row in trials if row["objective_change"] < 0.0]
    accepted = [row for row in trials if row["accepted"]]
    overshooting = [row for row in trials if row["objective_change"] > 0.0]
    finite = [row for row in trials if np.isfinite(row["objective_change"])]
    best = min(finite, key=lambda row: row["objective"]) if finite else None
    return {
        "largest_decreasing_step": max((row["step"] for row in decreasing), default=None),
        "largest_accepted_step": max((row["step"] for row in accepted), default=None),
        "smallest_overshooting_step": min((row["step"] for row in overshooting), default=None),
        "largest_overshooting_step": max((row["step"] for row in overshooting), default=None),
        "best_step": best["step"] if best else None,
        "best_objective": best["objective"] if best else None,
        "best_objective_change": best["objective_change"] if best else None,
        "num_decreasing_steps": len(decreasing),
        "num_accepted_steps": len(accepted),
        "num_overshooting_steps": len(overshooting),
    }


def all_target_blocks(model: RankOptimizationModel) -> tuple[str, ...]:
    """Return every optimized variable in model order.

    Model order is useful when comparing JSON output to the optimizer state:
    each fitted field contributes P, Q, and s, and the two scalar rates are
    appended at the end.
    """

    blocks: list[str] = []
    for core in model.cores:
        blocks.extend([core.p_key, core.q_key, core.s_key])
    blocks.extend(["cl", "cw", "rat"])
    return tuple(blocks)


def variable_shape(value: np.ndarray | float) -> list[int]:
    """JSON-safe variable shape."""

    return list(value.shape) if isinstance(value, np.ndarray) else []


def variable_norm_value(value: np.ndarray | float) -> float:
    """Euclidean norm for an array or scalar variable."""

    return float(np.linalg.norm(value)) if isinstance(value, np.ndarray) else abs(float(value))


def variable_max_abs(value: np.ndarray | float) -> float:
    """Maximum absolute entry for an array or scalar variable."""

    if isinstance(value, np.ndarray):
        return float(np.max(np.abs(value))) if value.size else 0.0
    return abs(float(value))


def core_and_side(model: RankOptimizationModel, block: str) -> tuple[RankCore, str]:
    """Split a rank block name such as ``omega_P`` into core and side."""

    name, side = block.split("_", 1)
    return model.core_by_name[name], side


def block_side(block: str) -> str | None:
    """Return P/Q/s for rank blocks, or None for scalar variables."""

    if "_" not in block:
        return None
    _, side = block.split("_", 1)
    return side


def basis_for_block(core: RankCore, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the value and derivative bases that localize P or Q rows."""

    if side == "P":
        return core.x0, core.x1
    if side == "Q":
        return core.y0, core.y1
    raise ValueError(f"unsupported side: {side}")


def axis_for_side(model: RankOptimizationModel, side: str) -> np.ndarray:
    """Return x1 for P blocks and x2 for Q blocks."""

    if side == "P":
        return model.x1
    if side == "Q":
        return model.x2
    raise ValueError(f"unsupported side: {side}")


def coefficient_row_support(
    basis: np.ndarray,
    derivative_basis: np.ndarray,
    axis: np.ndarray,
    row: int,
) -> dict[str, Any]:
    """Map one coefficient row to the grid region where its basis is active.

    P/Q row bands are meant to be spatially meaningful.  The support centroid
    and max-basis coordinate let the JSON explain which physical part of the
    grid each safe-step measurement corresponds to.
    """

    values = np.abs(basis[:, row])
    deriv_values = np.abs(derivative_basis[:, row])
    threshold = max(float(np.max(values)) * 1.0e-12, 1.0e-14)
    nz = np.flatnonzero(values > threshold)
    if nz.size == 0:
        nz = np.array([int(np.argmax(values))])
    weights = values[nz]
    weight_sum = float(np.sum(weights))
    centroid = float(np.sum(axis[nz] * weights) / weight_sum) if weight_sum else float(axis[nz[0]])
    max_index = int(np.argmax(values))
    return {
        "row": row,
        "support_index_min": int(nz[0]),
        "support_index_max": int(nz[-1]),
        "support_axis_min": float(axis[nz[0]]),
        "support_axis_max": float(axis[nz[-1]]),
        "basis_max_index": max_index,
        "basis_max_axis": float(axis[max_index]),
        "basis_centroid_axis": centroid,
        "basis_max_abs": float(np.max(values)),
        "derivative_max_abs": float(np.max(deriv_values)),
    }


def row_support_table(
    model: RankOptimizationModel,
    block: str,
    gradient: VariableDict,
    top_count: int = 16,
) -> dict[str, Any]:
    """Describe row localization and high-gradient rows for one variable block."""

    side = block_side(block)
    if side not in ("P", "Q"):
        value = gradient[block]
        return {
            "block": block,
            "side": side,
            "axis": None,
            "shape": variable_shape(value),
            "gradient_norm": variable_norm_value(value),
            "gradient_max_abs": variable_max_abs(value),
            "note": "non-spatial variable; probed as one block",
        }

    core, side = core_and_side(model, block)
    basis, derivative_basis = basis_for_block(core, side)
    axis = axis_for_side(model, side)
    block_gradient = gradient[block]
    assert isinstance(block_gradient, np.ndarray)
    row_norms = np.linalg.norm(block_gradient, axis=1)
    total_sq = float(np.sum(row_norms**2))
    rows = [
        {
            **coefficient_row_support(basis, derivative_basis, axis, row),
            "gradient_row_norm": float(row_norms[row]),
            "gradient_l2_mass_fraction_of_block": float(
                row_norms[row] ** 2 / max(total_sq, np.finfo(float).tiny)
            ),
        }
        for row in range(block_gradient.shape[0])
    ]
    top_rows = sorted(rows, key=lambda item: item["gradient_row_norm"], reverse=True)[:top_count]
    return {
        "block": block,
        "side": side,
        "axis": "x1" if side == "P" else "x2",
        "num_rows": int(block_gradient.shape[0]),
        "row_gradient_total_norm": float(np.sqrt(total_sq)),
        "top_gradient_rows": top_rows,
        "row_support_examples": [
            rows[index]
            for index in sorted(set([0, 1, 2, 3, 10, 100, 400, 700, len(rows) - 2, len(rows) - 1]))
            if 0 <= index < len(rows)
        ],
    }


def row_groups(
    model: RankOptimizationModel,
    block: str,
    gradient: VariableDict,
    edges: list[float],
) -> list[dict[str, Any]]:
    """Group a variable block into spatial row bands or one non-spatial block.

    P and Q rows are grouped by the coordinate where their basis is largest.
    Singular-value vectors and scalar rates do not have spatial rows, so they
    are intentionally probed as a single unit.
    """

    side = block_side(block)
    block_gradient = gradient[block]
    if side not in ("P", "Q"):
        norm = variable_norm_value(block_gradient)
        if isinstance(block_gradient, np.ndarray):
            row_count = int(block_gradient.shape[0]) if block_gradient.ndim else 1
            rows = list(range(row_count))
        else:
            row_count = 1
            rows = None
        return [
            {
                "label": f"{block} all entries",
                "row_min": 0,
                "row_max": row_count - 1,
                "row_count": row_count,
                "axis_min": None,
                "axis_max": None,
                "axis_mean": None,
                "effective_axis_gradient_weighted": None,
                "effective_log1p_axis2_gradient_weighted": None,
                "gradient_norm": norm,
                "gradient_l2_mass_fraction_of_block": 1.0 if norm > 0.0 else 0.0,
                "rows": rows,
                "spatial": False,
            }
        ]

    core, side = core_and_side(model, block)
    basis, derivative_basis = basis_for_block(core, side)
    axis = axis_for_side(model, side)
    block_gradient = gradient[block]
    assert isinstance(block_gradient, np.ndarray)
    row_norms = np.linalg.norm(block_gradient, axis=1)
    supports = [
        coefficient_row_support(basis, derivative_basis, axis, row)
        for row in range(block_gradient.shape[0])
    ]
    coords = np.array([item["basis_max_axis"] for item in supports], dtype=float)
    group_edges = [-np.inf, *edges, np.inf]
    groups: list[dict[str, Any]] = []
    total_sq = float(np.sum(row_norms**2))
    for index in range(len(group_edges) - 1):
        lo = group_edges[index]
        hi = group_edges[index + 1]
        if np.isneginf(lo):
            mask = coords <= hi
            label = f"{coords_name(side)}<= {hi:g}"
        elif np.isposinf(hi):
            mask = coords > lo
            label = f"{coords_name(side)}> {lo:g}"
        else:
            mask = (coords > lo) & (coords <= hi)
            label = f"{lo:g}< {coords_name(side)}<= {hi:g}"
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        weights = row_norms[rows] ** 2
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            effective_log_radius = float(
                np.sum(weights * np.log1p(coords[rows] ** 2)) / weight_sum
            )
            effective_axis = float(
                np.sum(weights * coords[rows]) / weight_sum
            )
        else:
            effective_log_radius = float(np.mean(np.log1p(coords[rows] ** 2)))
            effective_axis = float(np.mean(coords[rows]))
        groups.append(
            {
                "label": label,
                "row_min": int(rows[0]),
                "row_max": int(rows[-1]),
                "row_count": int(rows.size),
                "axis_min": float(np.min(coords[rows])),
                "axis_max": float(np.max(coords[rows])),
                "axis_mean": float(np.mean(coords[rows])),
                "effective_axis_gradient_weighted": effective_axis,
                "effective_log1p_axis2_gradient_weighted": effective_log_radius,
                "gradient_norm": float(np.sqrt(weight_sum)),
                "gradient_l2_mass_fraction_of_block": float(
                    weight_sum / max(total_sq, np.finfo(float).tiny)
                ),
                "rows": [int(row) for row in rows],
                "spatial": True,
            }
        )
    return groups


def coords_name(side: str) -> str:
    """Human-readable coordinate name for a P/Q side."""

    return "x1" if side == "P" else "x2"


def grouped_direction(
    variables: VariableDict,
    gradient: VariableDict,
    block: str,
    rows: list[int] | None,
) -> VariableDict:
    """Create the raw negative-gradient direction for one group only."""

    direction = zero_like_variables(variables)
    block_gradient = gradient[block]
    if isinstance(block_gradient, np.ndarray):
        block_direction = np.zeros_like(block_gradient)
        if rows is None:
            block_direction = -block_gradient.copy()
        elif block_gradient.ndim == 1:
            block_direction[rows] = -block_gradient[rows]
        else:
            block_direction[rows, :] = -block_gradient[rows, :]
        direction[block] = block_direction
    else:
        direction[block] = -float(block_gradient)
    return direction


def probe_group(
    model: RankOptimizationModel,
    variables: VariableDict,
    gradient: VariableDict,
    block: str,
    group: dict[str, Any],
    base_objective: float,
    base_components: dict[str, float],
    steps: list[float],
) -> dict[str, Any]:
    """Probe one row-band/block direction over the configured step ladder.

    This uses the same candidate evaluation as the optimizer: apply the trial
    step, retract/refit, evaluate residuals, and enforce gauge/decrease checks.
    The measured largest accepted step is therefore directly usable as a
    row-band scale in the optimizer.
    """

    raw_direction = grouped_direction(variables, gradient, block, group["rows"])
    projected = model.projected_tangent_direction(variables, raw_direction)
    direction, norm_before_normalization = normalize_direction(projected)
    directional_derivative = variable_dot(gradient, direction)
    trials: list[dict[str, Any]] = []
    if norm_before_normalization == 0.0:
        return {
            **{key: value for key, value in group.items() if key != "rows"},
            "direction_norm_before_normalization": norm_before_normalization,
            "directional_derivative": directional_derivative,
            "trial_summary": summarize_trials(trials),
            "trials": trials,
        }
    for step in steps:
        summary, _, evaluation = candidate_summary_with_evaluation(
            model,
            variables,
            direction,
            step,
            base_objective,
        )
        components = objective_components(model, evaluation.residuals)
        summary["objective_components"] = components
        summary["objective_component_changes"] = {
            key: components[key] - base_components[key] for key in components
        }
        summary["residual_rms"] = residual_rms_summary(model, evaluation)
        summary["passes_gauge"] = bool(max_abs(summary["gauge_errors"]) <= GAUGE_TOLERANCE)
        summary["passes_min_decrease"] = bool(summary["objective_change"] < -MIN_OBJECTIVE_DECREASE)
        trials.append(summary)

    trial_summary = summarize_trials(trials)
    largest_ok = trial_summary["largest_accepted_step"]
    high_over = None
    if largest_ok is not None:
        higher_overs = [
            row for row in trials
            if row["step"] > largest_ok and row["objective_change"] > 0.0
        ]
        if higher_overs:
            high_over = min(row["step"] for row in higher_overs)
    trial_summary["first_high_overshoot_step"] = high_over
    return {
        **{key: value for key, value in group.items() if key != "rows"},
        "direction_norm_before_normalization": norm_before_normalization,
        "directional_derivative": directional_derivative,
        "trial_summary": trial_summary,
        "trials": trials,
    }


def fit_power_law(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit a rough coordinate power law for spatial groups.

    The optimizer uses measured group scales directly; this fit is diagnostic
    metadata that helped reveal whether safe step sizes vary systematically
    with distance along x1 or x2.
    """

    usable = [
        group
        for group in groups
        if group.get("spatial", False)
        if group["trial_summary"]["largest_accepted_step"] is not None
        and group["gradient_l2_mass_fraction_of_block"] > 0.0
    ]
    if len(usable) < 2:
        return {"status": "insufficient_spatial_groups"}
    x = np.array([group["effective_log1p_axis2_gradient_weighted"] for group in usable])
    y = np.log(np.array([group["trial_summary"]["largest_accepted_step"] for group in usable]))
    w = np.array([group["gradient_l2_mass_fraction_of_block"] for group in usable])
    w = w / np.sum(w)
    x_bar = float(np.sum(w * x))
    y_bar = float(np.sum(w * y))
    var = float(np.sum(w * (x - x_bar) ** 2))
    if var <= 0.0:
        return {"status": "degenerate_axis"}
    alpha = float(np.sum(w * (x - x_bar) * (y - y_bar)) / var)
    log_c = float(y_bar - alpha * x_bar)
    predicted = log_c + alpha * x
    rmse = float(np.sqrt(np.sum(w * (predicted - y) ** 2)))
    return {
        "status": "ok",
        "fit": "largest_accepted_step ~= C * (1 + axis^2)^alpha",
        "alpha": alpha,
        "C": float(np.exp(log_c)),
        "weighted_log_rmse": rmse,
        "used_groups": [
            {
                "label": group["label"],
                "effective_axis": group["effective_axis_gradient_weighted"],
                "largest_accepted_step": group["trial_summary"]["largest_accepted_step"],
                "gradient_mass": group["gradient_l2_mass_fraction_of_block"],
            }
            for group in usable
        ],
    }


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constraint-weight", type=float, default=CONSTRAINT_WEIGHT)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--axis-edges", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = RankOptimizationModel(DATA_PATH, constraint_weight=args.constraint_weight)
    variables = model.complete_variables(load_state(args.state))
    evaluation = model.evaluate(variables)
    gradient = model.analytic_gradient(variables, evaluation=evaluation)
    base_objective = model.objective_from_residuals(evaluation.residuals)
    base_components = objective_components(model, evaluation.residuals)
    steps = parse_steps(args.steps)
    edges = parse_edges(args.axis_edges)
    target_blocks = all_target_blocks(model)

    block_results: dict[str, Any] = {}
    for block in target_blocks:
        support = row_support_table(model, block, gradient)
        groups = row_groups(model, block, gradient, edges)
        group_probes = [
            probe_group(
                model,
                variables,
                gradient,
                block,
                group,
                base_objective,
                base_components,
                steps,
            )
            for group in groups
            if group["gradient_norm"] > 0.0
        ]
        block_results[block] = {
            "support": support,
            "groups": group_probes,
            "power_law_fit": fit_power_law(group_probes),
        }

    output = {
        "state": str(args.state),
        "constraint_weight": args.constraint_weight,
        "objective": base_objective,
        "objective_components": base_components,
        "residual_rms": residual_rms_summary(model, evaluation),
        "gradient_blocks": variable_block_summary(gradient),
        "steps": steps,
        "axis_edges": edges,
        "target_blocks": target_blocks,
        "block_results": block_results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"objective {base_objective:.12e}")
    for block, result in block_results.items():
        fit = result["power_law_fit"]
        if fit.get("status") == "ok":
            fit_text = f"alpha={fit['alpha']:.3f} C={fit['C']:.3e}"
        else:
            fit_text = fit.get("status", "unknown")
        print(f"{block}: {fit_text}")
        for group in result["groups"]:
            summary = group["trial_summary"]
            print(
                f"  {group['label']:18s} mass={group['gradient_l2_mass_fraction_of_block']:.3e} "
                f"ok={summary['largest_accepted_step']} over={summary['first_high_overshoot_step']}"
            )


if __name__ == "__main__":
    main()
