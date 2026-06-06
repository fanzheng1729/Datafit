"""Shared explicit-S rank-factor optimization model for Datafit.

This module is the common implementation used by the gradient checker,
retraction checker, and future rank-factor optimizer.  The optimized cores use

    core = L diag(s) M.T

with B-spline coefficient variables ``P, Q`` and a separate amplitude vector
``s`` for each fitted field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np

import runfit
from rank_factor_tools import (
    retract_and_refit,
    value_gradient,
    weighted_orthonormality_error,
)


# ``data.mat`` is the single source of fitted profile data.  Keeping the path
# relative to this file lets every command-line entry point run from any cwd.
DATA_PATH = Path(__file__).with_name("data.mat")

# The pushed row-band result uses this conservative low constraint weight.  The
# value came from the P/Q gradient-balance sweep: it is small enough for the
# fitted equations to move, but still kept divergence and curl nonincreasing in
# the recorded 30-step run.
CONSTRAINT_WEIGHT = 3.0e-4


VariableDict = dict[str, np.ndarray | float]


@dataclass
class RankCore:
    """Static B-spline/SVD data for one optimized field.

    Each physical field is represented as ``x_factor * (P S Q.T)`` plus, for
    velocity, a fixed far-field contribution.  ``x0/x1`` and ``y0/y1`` are the
    value and first-derivative B-spline matrices used to evaluate the current
    coefficient variables and to pull adjoints back to coefficient space.
    """

    name: str
    parity: int
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    p: np.ndarray
    q: np.ndarray
    singular_values: np.ndarray
    rank: int

    @property
    def p_key(self) -> str:
        return f"{self.name}_P"

    @property
    def q_key(self) -> str:
        return f"{self.name}_Q"

    @property
    def s_key(self) -> str:
        return f"{self.name}_s"


@dataclass
class Evaluation:
    """One full model evaluation cached at both field and residual levels."""

    objective: float
    fields: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]
    core_cache: dict[str, dict[str, np.ndarray]]


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(value * value)))


def relative_norm(error: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(error) / denominator)


def copy_variables(variables: VariableDict) -> VariableDict:
    """Deep-copy the mixed array/scalar variable dictionary."""

    out: VariableDict = {}
    for key, value in variables.items():
        out[key] = value.copy() if isinstance(value, np.ndarray) else float(value)
    return out


def variable_dot(left: VariableDict, right: VariableDict) -> float:
    """Euclidean dot product in the current coefficient/scalar coordinates."""

    total = 0.0
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, np.ndarray):
            assert isinstance(right_value, np.ndarray)
            total += float(np.sum(left_value * right_value))
        else:
            total += float(left_value) * float(right_value)
    return total


def variable_norm(variables: VariableDict) -> float:
    return float(np.sqrt(max(variable_dot(variables, variables), 0.0)))


def add_scaled_variables(base: VariableDict, direction: VariableDict, scale: float) -> VariableDict:
    """Return ``base + scale * direction`` without mutating scalar entries."""

    out = copy_variables(base)
    for key, value in out.items():
        direction_value = direction[key]
        if isinstance(value, np.ndarray):
            assert isinstance(direction_value, np.ndarray)
            value += scale * direction_value
        else:
            out[key] = float(value) + scale * float(direction_value)
    return out


def build_rank_core(
    name: str,
    core: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    parity: int,
    eps_svd: float,
) -> RankCore:
    """Build explicit-S B-spline coefficients for a fitted core.

    The saved profile stores evaluated fields on the mesh.  The optimizer
    works in B-spline coefficient space, so initialization first performs an
    SVD of the mesh field and then solves for B-spline coefficients that
    reproduce the left and right singular vectors.
    """

    u, singular_values, vh = np.linalg.svd(core, full_matrices=False)
    v = vh.T
    rank = runfit.svd_order(u, singular_values, v, eps_svd)
    u = u[:, :rank]
    v = v[:, :rank]
    singular_values = singular_values[:rank]
    if parity == 1:
        # Odd-reflected fields vanish at x1=0.  Dropping the origin row mirrors
        # the MATLAB fitting convention and prevents the coefficient solve from
        # fighting an enforced zero boundary value.
        u[0, :] = 0.0

    basis_x, basis_y = runfit.bs6mat(x1, x2, 2, parity)
    if parity == 1:
        p = np.linalg.solve(basis_x[0][1:], u[1:])
    else:
        p = np.linalg.solve(basis_x[0], u)
    q = np.linalg.solve(basis_y[0], v)

    return RankCore(
        name=name,
        parity=parity,
        x0=basis_x[0],
        x1=basis_x[1],
        y0=basis_y[0],
        y1=basis_y[1],
        p=p,
        q=q,
        singular_values=singular_values,
        rank=rank,
    )


class RankOptimizationModel:
    """Objective, gradients, gauges, and retraction for explicit-S variables.

    This class owns the mathematical contract used by every optimizer script:
    a variable dictionary is converted to physical fields, the four residual
    equations are evaluated on the grid, and analytic adjoints are pushed back
    to the low-rank B-spline coefficients.  Keeping this in one place makes the
    vanilla, row-band, and diagnostic scripts comparable.
    """

    def __init__(
        self,
        data_path: Path = DATA_PATH,
        normalize_initial: bool = True,
        constraint_weight: float = CONSTRAINT_WEIGHT,
    ):
        if constraint_weight <= 0.0:
            raise ValueError("constraint_weight must be positive")
        self.constraint_weight = float(constraint_weight)

        # Load only numerical data from the MAT file.  The MATLAB file contains
        # function handles too, but ``runfit.load_numeric_data`` deliberately
        # avoids requiring MATLAB to deserialize them.
        data = runfit.load_numeric_data(data_path)
        self.x1 = np.asarray(data["x1"], dtype=float)
        self.x2 = np.asarray(data["x2"], dtype=float)
        self.x1_col = self.x1[:, None]
        self.x2_row = self.x2[None, :]
        self.n_grid = self.x1.size * self.x2.size

        omega = np.asarray(data["w"], dtype=float)
        zeta = np.asarray(data["v"], dtype=float)
        vel = data["Vel"]
        u10f = np.asarray(vel["u10f"], dtype=float)
        u20f = np.asarray(vel["u20f"], dtype=float)

        # The rank factors are fitted to rescaled cores, not directly to the
        # physical fields.  These analytic weights restore the physical
        # profiles and provide their product-rule derivatives.
        self.profile_factor = np.sqrt(1.0 + self.x1_col**2) / np.sqrt(
            1.0 + self.x1_col**2 + self.x2_row**2
        )
        self.profile_factor_x1 = self.profile_factor * (
            self.x1_col
            * self.x2_row**2
            / (1.0 + self.x1_col**2)
            / (1.0 + self.x1_col**2 + self.x2_row**2)
        )
        self.profile_factor_x2 = self.profile_factor * (
            -self.x2_row / (1.0 + self.x1_col**2 + self.x2_row**2)
        )

        # The near-field velocity factors use one-dimensional damping weights.
        # The far-field velocity is not optimized; it is reconstructed once in
        # ``_build_fixed_velocity`` and then added to every evaluation.
        factor0_u1 = 1.0 + self.x1_col**2
        self.u1_factor = factor0_u1**0.25
        self.u1_factor_x1 = 0.5 * self.x1_col * self.u1_factor / factor0_u1
        factor0_u2 = 1.0 + self.x2_row**2
        self.u2_factor = factor0_u2**0.25
        self.u2_factor_x2 = 0.5 * self.x2_row * self.u2_factor / factor0_u2

        # The SVD cutoffs follow the older fitting scripts: omega/zeta are
        # slightly less strict than the two velocity components.
        self.cores = [
            build_rank_core("omega", omega / self.profile_factor, self.x1, self.x2, 1, 1.0e-10),
            build_rank_core("zeta", zeta / self.profile_factor, self.x1, self.x2, 1, 1.0e-10),
            build_rank_core("u1", u10f / self.u1_factor, self.x1, self.x2, 1, 1.0e-11),
            build_rank_core("u2", u20f / self.u2_factor, self.x1, self.x2, 0, 1.0e-11),
        ]
        self.core_by_name = {core.name: core for core in self.cores}

        self.fixed_velocity = self._build_fixed_velocity(data)
        omega_x1 = np.asarray(data["wx1"], dtype=float)
        zeta_x1 = np.asarray(data["vx1"], dtype=float)
        # The two scalar rates are initialized from the origin identities used
        # in the original dynamic-rescaling formulation.
        cl = float(4.0 * zeta_x1[0, 0] / omega_x1[0, 0])
        cw = float(vel["u1dx1"][0, 0] + cl / 2.0)

        self.variables: VariableDict = {"cl": cl, "cw": cw}
        for core in self.cores:
            self.variables[core.p_key] = core.p.copy()
            self.variables[core.q_key] = core.q.copy()
            self.variables[core.s_key] = core.singular_values.copy()
        if normalize_initial:
            self.variables = self.retract_variables(self.variables, force=True)

        # Gauges and residual scales are frozen at the normalized initial chart.
        # The objective then measures relative movement from that baseline.
        base = self.evaluate(self.variables)
        self.gauge_targets = self.gauge_values_from_fields(base.fields)
        self.scales = {name: max(rms(value), np.finfo(float).tiny) for name, value in base.residuals.items()}

    def array_keys(self) -> list[str]:
        keys: list[str] = []
        for core in self.cores:
            keys.extend([core.p_key, core.q_key, core.s_key])
        return keys

    def variable_keys(self) -> list[str]:
        return self.array_keys() + ["cl", "cw"]

    def _build_fixed_velocity(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        """Reconstruct the non-optimized far-field velocity contribution.

        The optimized ``u1``/``u2`` cores cover only the near-field pieces.  The
        saved streamfunction correction ``Psi1`` contributes a fixed velocity
        tail and its derivatives, which must be present for the curl and
        divergence residuals to match the saved MATLAB fit.
        """

        polar_coeff = runfit.xycoef(
            np.asarray(data["gx1"], dtype=float),
            np.asarray(data["gx2"], dtype=float),
            float(data["alpha_b"]),
            order=3,
        )
        psi1 = runfit.deri_psi1(
            len(data["gx1"]),
            len(data["gx2"]),
            np.asarray(data["p_ag_coe"], dtype=float),
            np.asarray(data["BS1d_large"], dtype=object),
            polar_coeff,
            order=2,
        )
        rat = float(np.asarray(data["rec"])[6])
        n1 = self.x1.size
        n2 = self.x2.size
        return {
            "u1": -rat * psi1[(0, 1)][:n1, :n2],
            "u1x1": -rat * psi1[(1, 1)][:n1, :n2],
            "u1x2": -rat * psi1[(0, 2)][:n1, :n2],
            "u2": rat * psi1[(1, 0)][:n1, :n2],
            "u2x1": rat * psi1[(2, 0)][:n1, :n2],
            "u2x2": rat * psi1[(1, 1)][:n1, :n2],
        }

    def _core_eval(self, core: RankCore, variables: VariableDict) -> dict[str, np.ndarray]:
        """Evaluate one core and its first derivatives on the tensor grid."""

        p = variables[core.p_key]
        q = variables[core.q_key]
        s = variables[core.s_key]
        assert isinstance(p, np.ndarray)
        assert isinstance(q, np.ndarray)
        assert isinstance(s, np.ndarray)
        left = core.x0 @ p
        left_x1 = core.x1 @ p
        right = core.y0 @ q
        right_x2 = core.y1 @ q
        value, x1, x2 = value_gradient(left, left_x1, right, right_x2, s)
        return {
            "left": left,
            "left_x1": left_x1,
            "right": right,
            "right_x2": right_x2,
            "s": s,
            "value": value,
            "x1": x1,
            "x2": x2,
        }

    def evaluate(self, variables: VariableDict) -> Evaluation:
        """Evaluate physical fields, residual equations, and cached core data."""

        core_cache = {core.name: self._core_eval(core, variables) for core in self.cores}
        omega_core = core_cache["omega"]
        zeta_core = core_cache["zeta"]
        u1_core = core_cache["u1"]
        u2_core = core_cache["u2"]

        # Product-rule restoration from rescaled rank cores to physical fields.
        # The rank core derivatives are derivatives of the fitted core; the
        # analytic factors contribute the remaining derivative terms.
        omega = self.profile_factor * omega_core["value"]
        omega_x1 = self.profile_factor * omega_core["x1"] + self.profile_factor_x1 * omega_core["value"]
        omega_x2 = self.profile_factor * omega_core["x2"] + self.profile_factor_x2 * omega_core["value"]
        zeta = self.profile_factor * zeta_core["value"]
        zeta_x1 = self.profile_factor * zeta_core["x1"] + self.profile_factor_x1 * zeta_core["value"]
        zeta_x2 = self.profile_factor * zeta_core["x2"] + self.profile_factor_x2 * zeta_core["value"]

        # Velocity is split into optimized near-field rank cores and a fixed
        # far-field streamfunction-derived piece.
        u1 = self.u1_factor * u1_core["value"] + self.fixed_velocity["u1"]
        u1x1 = (
            self.u1_factor * u1_core["x1"]
            + self.u1_factor_x1 * u1_core["value"]
            + self.fixed_velocity["u1x1"]
        )
        u1x2 = self.u1_factor * u1_core["x2"] + self.fixed_velocity["u1x2"]
        u2 = self.u2_factor * u2_core["value"] + self.fixed_velocity["u2"]
        u2x1 = self.u2_factor * u2_core["x1"] + self.fixed_velocity["u2x1"]
        u2x2 = (
            self.u2_factor * u2_core["x2"]
            + self.u2_factor_x2 * u2_core["value"]
            + self.fixed_velocity["u2x2"]
        )

        cl = float(variables["cl"])
        cw = float(variables["cw"])
        # The four residuals are the dependent equations solved by the fit:
        # two profile equations plus incompressibility and curl consistency.
        fomega = runfit.omega_residual(
            cl, cw, self.x1, self.x2, omega, zeta, omega_x1, omega_x2, u1, u2, zeta_x1
        )
        fzeta = runfit.zeta_residual(cl, cw, self.x1, self.x2, zeta, zeta_x1, zeta_x2, u1, u2)
        divergence = u1x1 + u2x2
        curl = u1x2 - u2x1 - omega
        residuals = {
            "fomega": fomega,
            "fzeta": fzeta,
            "divergence": divergence,
            "curl": curl,
        }
        fields = {
            "omega": omega,
            "omega_x1": omega_x1,
            "omega_x2": omega_x2,
            "zeta": zeta,
            "zeta_x1": zeta_x1,
            "zeta_x2": zeta_x2,
            "u1": u1,
            "u1x1": u1x1,
            "u1x2": u1x2,
            "u2": u2,
            "u2x1": u2x1,
            "u2x2": u2x2,
        }
        objective = 0.0
        if hasattr(self, "scales"):
            objective = self.objective_from_residuals(residuals)
        return Evaluation(objective=objective, fields=fields, residuals=residuals, core_cache=core_cache)

    def objective_from_residuals(self, residuals: dict[str, np.ndarray]) -> float:
        """Return the weighted mean-square objective from residual arrays."""

        return float(
            0.5 * np.mean((residuals["fomega"] / self.scales["fomega"]) ** 2)
            + 0.5 * np.mean((residuals["fzeta"] / self.scales["fzeta"]) ** 2)
            + 0.5 * self.constraint_weight * np.mean((residuals["divergence"] / self.scales["divergence"]) ** 2)
            + 0.5 * self.constraint_weight * np.mean((residuals["curl"] / self.scales["curl"]) ** 2)
        )

    def objective(self, variables: VariableDict) -> float:
        return self.objective_from_residuals(self.evaluate(variables).residuals)

    def residual_rms_from_residuals(self, residuals: dict[str, np.ndarray]) -> dict[str, float]:
        return {name: rms(value) for name, value in residuals.items()}

    def residual_rms(self, variables: VariableDict) -> dict[str, float]:
        return self.residual_rms_from_residuals(self.evaluate(variables).residuals)

    def gauge_values_from_fields(self, fields: dict[str, np.ndarray]) -> dict[str, float]:
        """Return the two origin gauges kept fixed by tangent projection."""

        return {
            "omega_x1_00": float(fields["omega_x1"][0, 0]),
            "theta_x1x1_00": float(2.0 * fields["zeta_x1"][0, 0]),
        }

    def gauge_values(self, variables: VariableDict) -> dict[str, float]:
        return self.gauge_values_from_fields(self.evaluate(variables).fields)

    def gauge_errors_from_fields(self, fields: dict[str, np.ndarray]) -> dict[str, float]:
        values = self.gauge_values_from_fields(fields)
        return {key: values[key] - self.gauge_targets[key] for key in values}

    def gauge_errors(self, variables: VariableDict) -> dict[str, float]:
        return self.gauge_errors_from_fields(self.evaluate(variables).fields)

    def _lambdas(self, residuals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Residual adjoints for the normalized least-squares objective."""

        n = self.n_grid
        return {
            "fomega": residuals["fomega"] / (self.scales["fomega"] ** 2 * n),
            "fzeta": residuals["fzeta"] / (self.scales["fzeta"] ** 2 * n),
            "divergence": self.constraint_weight * residuals["divergence"] / (self.scales["divergence"] ** 2 * n),
            "curl": self.constraint_weight * residuals["curl"] / (self.scales["curl"] ** 2 * n),
        }

    def analytic_gradient(
        self,
        variables: VariableDict,
        evaluation: Evaluation | None = None,
    ) -> VariableDict:
        """Return the analytic gradient in coefficient/scalar coordinates."""

        field_gradients = self.field_gradient(variables, evaluation=evaluation)
        evaluation = field_gradients["evaluation"]
        fields = evaluation.fields
        lam = self._lambdas(evaluation.residuals)
        cl = float(variables["cl"])
        a1 = cl * self.x1_col + fields["u1"]
        a2 = cl * self.x2_row + fields["u2"]

        # ``cl`` and ``cw`` enter only the two profile residuals, so their
        # gradients are direct contractions of those residual adjoints.
        gradient: VariableDict = {
            "cl": float(
                np.sum(lam["fomega"] * (-(self.x1_col * fields["omega_x1"] + self.x2_row * fields["omega_x2"])))
                + np.sum(lam["fzeta"] * (-(self.x1_col * fields["zeta_x1"] + self.x2_row * fields["zeta_x2"])))
            ),
            "cw": float(np.sum(lam["fomega"] * fields["omega"]) + np.sum(lam["fzeta"] * (2.0 * fields["zeta"]))),
        }

        self._add_rank_gradient(
            gradient, "omega", *field_gradients["omega"], evaluation.core_cache["omega"],
            self.profile_factor, self.profile_factor_x1, self.profile_factor_x2
        )
        self._add_rank_gradient(
            gradient, "zeta", *field_gradients["zeta"], evaluation.core_cache["zeta"],
            self.profile_factor, self.profile_factor_x1, self.profile_factor_x2
        )
        self._add_rank_gradient(
            gradient, "u1", *field_gradients["u1"], evaluation.core_cache["u1"],
            self.u1_factor, self.u1_factor_x1, np.zeros_like(self.u1_factor)
        )
        self._add_rank_gradient(
            gradient, "u2", *field_gradients["u2"], evaluation.core_cache["u2"],
            self.u2_factor, np.zeros_like(self.u2_factor), self.u2_factor_x2
        )
        return gradient

    def field_gradient(
        self,
        variables: VariableDict,
        evaluation: Evaluation | None = None,
    ) -> dict[str, Any]:
        """Return adjoints with respect to field values and derivatives.

        Each tuple is ``(dJ/d value, dJ/d x1_derivative, dJ/d x2_derivative)``
        for the named field before the rank-factor pullback.  Keeping this
        separately exposed made the gradient-localization diagnostics possible.
        """

        if evaluation is None:
            evaluation = self.evaluate(variables)
        fields = evaluation.fields
        lam = self._lambdas(evaluation.residuals)
        cl = float(variables["cl"])
        cw = float(variables["cw"])
        a1 = cl * self.x1_col + fields["u1"]
        a2 = cl * self.x2_row + fields["u2"]
        # The zeta equation contains ``u1/x1``.  At the symmetry axis the term
        # is omitted to avoid an artificial division by zero; the first mesh row
        # is already controlled by the gauge conditions.
        h_adjoint = np.zeros_like(fields["u1"])
        h_adjoint[1:] = (fields["zeta"] * lam["fzeta"])[1:] / self.x1[1:, None]
        return {
            "evaluation": evaluation,
            "omega": (
                cw * lam["fomega"] - lam["curl"],
                -a1 * lam["fomega"],
                -a2 * lam["fomega"],
            ),
            "zeta": (
                lam["fomega"] + (2.0 * cw - self._u1_over_x1(fields["u1"])) * lam["fzeta"],
                self.x1_col * lam["fomega"] - a1 * lam["fzeta"],
                -a2 * lam["fzeta"],
            ),
            "u1": (
                -fields["omega_x1"] * lam["fomega"] - fields["zeta_x1"] * lam["fzeta"] - h_adjoint,
                lam["divergence"],
                lam["curl"],
            ),
            "u2": (
                -fields["omega_x2"] * lam["fomega"] - fields["zeta_x2"] * lam["fzeta"],
                -lam["curl"],
                lam["divergence"],
            ),
        }

    def _u1_over_x1(self, u1: np.ndarray) -> np.ndarray:
        """Compute ``u1/x1`` with the axis row set to zero."""

        out = np.zeros_like(u1)
        out[1:] = u1[1:] / self.x1[1:, None]
        return out

    def _add_rank_gradient(
        self,
        gradient: VariableDict,
        name: str,
        g_value: np.ndarray,
        g_x1: np.ndarray,
        g_x2: np.ndarray,
        cache: dict[str, np.ndarray],
        factor: np.ndarray,
        factor_x1: np.ndarray,
        factor_x2: np.ndarray,
    ) -> None:
        """Pull field adjoints back through one low-rank B-spline core."""

        core = self.core_by_name[name]
        # Undo the physical scaling in adjoint form.  A derivative residual can
        # hit both the rank-core derivative and the derivative of the analytic
        # profile factor, hence the three core adjoints below.
        g_core = factor * g_value + factor_x1 * g_x1 + factor_x2 * g_x2
        g_core_x1 = factor * g_x1
        g_core_x2 = factor * g_x2
        left = cache["left"]
        left_x1 = cache["left_x1"]
        right = cache["right"]
        right_x2 = cache["right_x2"]
        s = cache["s"]

        # The identities are the matrix-calculus adjoints of
        # ``left @ diag(s) @ right.T`` and its x1/x2 derivative variants.
        right_s = right * s[None, :]
        right_x2_s = right_x2 * s[None, :]
        left_s = left * s[None, :]
        left_x1_s = left_x1 * s[None, :]
        gradient[core.p_key] = (
            core.x0.T @ (g_core @ right_s)
            + core.x1.T @ (g_core_x1 @ right_s)
            + core.x0.T @ (g_core_x2 @ right_x2_s)
        )
        gradient[core.q_key] = (
            core.y0.T @ (g_core.T @ left_s)
            + core.y0.T @ (g_core_x1.T @ left_x1_s)
            + core.y1.T @ (g_core_x2.T @ left_s)
        )
        gradient[core.s_key] = (
            np.sum(left * (g_core @ right), axis=0)
            + np.sum(left_x1 * (g_core_x1 @ right), axis=0)
            + np.sum(left * (g_core_x2 @ right_x2), axis=0)
        )

    def gauge_gradients(self, variables: VariableDict) -> dict[str, VariableDict]:
        """Return coefficient gradients of the two fixed gauge values."""

        zero = self.zero_like_variables(variables)
        return {
            "omega_x1_00": self._single_gauge_gradient("omega", variables, scale=1.0, template=zero),
            "theta_x1x1_00": self._single_gauge_gradient("zeta", variables, scale=2.0, template=zero),
        }

    def _single_gauge_gradient(
        self,
        name: str,
        variables: VariableDict,
        scale: float,
        template: VariableDict,
    ) -> VariableDict:
        """Gradient of one origin derivative gauge for omega or zeta."""

        core = self.core_by_name[name]
        cache = self._core_eval(core, variables)
        factor = self.profile_factor[0, 0]
        factor_x1 = self.profile_factor_x1[0, 0]
        left = cache["left"][0]
        left_x1 = cache["left_x1"][0]
        right = cache["right"][0]
        s = cache["s"]
        out = copy_variables(template)
        out[core.p_key] = scale * (
            np.outer(core.x1[0], factor * right * s)
            + np.outer(core.x0[0], factor_x1 * right * s)
        )
        out[core.q_key] = scale * np.outer(
            core.y0[0],
            factor * left_x1 * s + factor_x1 * left * s,
        )
        out[core.s_key] = scale * (factor * left_x1 * right + factor_x1 * left * right)
        return out

    def zero_like_variables(self, variables: VariableDict) -> VariableDict:
        """Return a zero variable dictionary with the same mixed structure."""

        out: VariableDict = {}
        for key, value in variables.items():
            out[key] = np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0
        return out

    def projected_tangent_direction(self, variables: VariableDict, raw_direction: VariableDict) -> VariableDict:
        """Project a direction onto the tangent space of the two fixed gauges.

        The optimizer is allowed to change the profile, but it should not drift
        in the two scalar normalizations inherited from the original fit.  This
        solves the small normal-equation system for the gauge-gradient
        correction and subtracts it from the raw direction.
        """

        gauges = self.gauge_gradients(variables)
        names = ["omega_x1_00", "theta_x1x1_00"]
        gram = np.array(
            [[variable_dot(gauges[a], gauges[b]) for b in names] for a in names],
            dtype=float,
        )
        rhs = np.array([variable_dot(gauges[name], raw_direction) for name in names], dtype=float)
        # The gauge Gram matrix is 2x2; the pseudo-inverse branch is a guard
        # against near-collinearity if a future state degenerates.
        if np.linalg.cond(gram) > 1.0e14:
            correction = np.linalg.pinv(gram) @ rhs
        else:
            correction = np.linalg.solve(gram, rhs)

        projected = copy_variables(raw_direction)
        for coefficient, name in zip(correction, names):
            gauge_gradient = gauges[name]
            for key, value in projected.items():
                gradient_value = gauge_gradient[key]
                if isinstance(value, np.ndarray):
                    assert isinstance(gradient_value, np.ndarray)
                    value -= coefficient * gradient_value
                else:
                    projected[key] = float(value) - coefficient * float(gradient_value)
        return projected

    def core_chart_errors(self, core: RankCore, variables: VariableDict) -> dict[str, float]:
        """Measure violations of the explicit-S chart normalization."""

        p = variables[core.p_key]
        q = variables[core.q_key]
        s = variables[core.s_key]
        assert isinstance(p, np.ndarray)
        assert isinstance(q, np.ndarray)
        assert isinstance(s, np.ndarray)
        left = core.x0 @ p
        right = core.y0 @ q
        # The retraction convention keeps columns orthonormal, singular values
        # nonnegative, and singular values sorted from largest to smallest.
        sorted_violation = 0.0
        if s.size > 1:
            sorted_violation = max(0.0, float(np.max(s[1:] - s[:-1])))
        negative_violation = max(0.0, float(-np.min(s))) if s.size else 0.0
        return {
            "left_gram_error": weighted_orthonormality_error(left),
            "right_gram_error": weighted_orthonormality_error(right),
            "singular_sort_violation": sorted_violation,
            "singular_negative_violation": negative_violation,
        }

    def core_chart_is_normalized(
        self,
        core: RankCore,
        variables: VariableDict,
        tolerance: float,
    ) -> bool:
        """Return whether a core is already close enough to retraction form."""

        errors = self.core_chart_errors(core, variables)
        return all(value <= tolerance for value in errors.values())

    def chart_error_summary(self, variables: VariableDict) -> dict[str, dict[str, float]]:
        """Return chart-normalization diagnostics for every optimized core."""

        return {
            core.name: self.core_chart_errors(core, variables)
            for core in self.cores
        }

    def retract_variables(
        self,
        variables: VariableDict,
        *,
        force: bool = False,
        tolerance: float = 1.0e-10,
    ) -> VariableDict:
        """Retract all rank cores back to the normalized explicit-S chart.

        Trial steps perturb the coefficient factors directly.  Retraction keeps
        the represented field nearly unchanged while removing gauge freedom in
        the low-rank factorization; without it, the same field could be encoded
        by many badly scaled ``P, Q, s`` triples.
        """

        out = copy_variables(variables)
        for core in self.cores:
            p = out[core.p_key]
            q = out[core.q_key]
            s = out[core.s_key]
            assert isinstance(p, np.ndarray)
            assert isinstance(q, np.ndarray)
            assert isinstance(s, np.ndarray)
            # Skipping already-normalized cores avoids unnecessary SVD/refit
            # noise after small steps, but initialization forces every core
            # through the chart once.
            if not force and self.core_chart_is_normalized(core, out, tolerance):
                continue
            left = core.x0 @ p
            right = core.y0 @ q
            result = retract_and_refit(left, right, s, core.x0, core.y0, core.parity)
            out[core.p_key] = result.left_coefficients
            out[core.q_key] = result.right_coefficients
            out[core.s_key] = result.singular_values
        return out
