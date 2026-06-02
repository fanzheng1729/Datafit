#!/usr/bin/env python3
"""Python equivalent of runfit.m using the notation of Analysis/Numerics.

The MAT file keeps the original short variable names:

* ``w`` is the paper's vorticity profile, omega.
* ``v`` is zeta = theta / x1, not a velocity component.
* ``Vel.u1`` and ``Vel.u2`` are the velocity components.

The residual checks are the steady dynamic-rescaling equations from
Analysis (2.10)-(2.11), with theta replaced by x1*zeta as in Numerics
Section 7 and Appendix C.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from local_deps import add_local_deps

add_local_deps(__file__)

import numpy as np

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        "runfit.py needs SciPy to read compressed MATLAB v5 data and sparse "
        "matrices. Install requirements.txt first."
    ) from exc


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(value) ** 2)))


def load_numeric_data(path: Path) -> dict[str, Any]:
    """Load MAT variables that have Python numeric equivalents.

    data.mat also contains MATLAB function handles named AG and Chi20. SciPy can
    deserialize their containers, but it cannot execute MATLAB function
    handles, so this translation rebuilds their formulas below.
    """

    names = [
        "BS1d_large",
        "Vel",
        "alpha_b",
        "gx1",
        "gx2",
        "p_ag_coe",
        "rec",
        "v",
        "vx1",
        "vx2",
        "w",
        "wx1",
        "wx2",
        "x1",
        "x2",
    ]
    raw = loadmat(path, variable_names=names, simplify_cells=True)
    return {name: raw[name] for name in names}


def meshext(x: np.ndarray, count: int, rate: float) -> np.ndarray:
    return x[-1] * rate ** np.arange(1, count + 1, dtype=float)


def bscoe(order: int, rows: int | None = None) -> np.ndarray:
    if rows is None:
        rows = order // 2 - 1
    out = np.empty((rows, order), dtype=float)
    for row, i in enumerate(range(rows - 1, -1, -1)):
        for col, j in enumerate(range(1, order + 1)):
            out[row, col] = (
                (-1) ** (j - 1)
                * math.comb(order + i, i + j)
                * math.comb(i + j - 1, i)
            )
    return out


def bs6n(
    x: np.ndarray | float,
    support: np.ndarray,
    degree: int,
    near_id: int,
    far_id: int | float,
    near_h: float,
    far_h: float,
    deriv_count: int,
) -> np.ndarray:
    """Evaluate the normalized B-spline basis from BS/BS6N.m."""

    z = np.atleast_1d(np.asarray(x, dtype=float))
    s = np.sort(np.asarray(support, dtype=float))
    if degree != s.size - 1:
        raise ValueError("B-spline support size does not match its degree")

    diff = s[None, :] - z[:, None]
    in_support = (z > s[0]) & (z < s[-1])
    use_large_formula = (z > s[degree // 2]) & (z < s[-1])
    use_small_formula = in_support.astype(float) - use_large_formula.astype(float)

    denom = np.empty_like(s)
    for index in range(degree + 1):
        factors = s[index] - s
        factors[index] = 1.0
        denom[index] = np.prod(factors)

    signs = (
        np.sign(np.maximum(diff, 0.0)) - use_small_formula[:, None]
    ) * in_support[:, None]

    if near_id <= 8:
        scale = near_h
    elif far_id <= 8:
        scale = far_h / 100.0
    else:
        mid = degree // 2
        scale = (s[mid + 1] - s[mid - 1]) / 2.0

    derivs = np.empty((deriv_count, z.size), dtype=float)
    for deriv in range(deriv_count):
        raw = np.sum(
            signs * diff ** (degree - 1 - deriv) / denom[None, :] * (-1) ** deriv,
            axis=1,
        )
        factor = math.prod(range(degree - deriv, degree + 1))
        derivs[deriv] = raw * scale * factor
    return derivs


def support_indices(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.flatnonzero((values - low) * (values - high) < 0.0)


def extended_knots(mesh: np.ndarray, rate: float, order: int) -> np.ndarray:
    return np.concatenate(
        (
            np.array([-mesh[3], -mesh[2], -mesh[1]], dtype=float),
            mesh,
            meshext(mesh, order - 2, rate),
        )
    )


def x_bs6_matrices(
    mesh: np.ndarray, values: np.ndarray, deriv_count: int, parity: int
) -> list[np.ndarray]:
    """Build x-direction BS6_interp or BS6_interp2 matrices for constant weights."""

    order = 7
    degree = order - 1
    n = mesh.size
    rate = mesh[-1] / mesh[-2]
    xx = extended_knots(mesh, rate, order)
    coeff = bscoe(order)
    h = mesh[1] - mesh[0]
    far_h = mesh[-1] - mesh[-2]
    odd = parity == 1
    basis_count = n - 1 if odd else n
    mats = np.zeros((deriv_count, values.size, basis_count), dtype=float)

    first_basis = 1 if odd else 0
    for basis in range(first_basis, n):
        knots = xx[basis : basis + order]
        row_ids = support_indices(values, knots[0], knots[-1])
        row_ids = row_ids[values[row_ids] != 0.0]
        if row_ids.size == 0:
            continue

        z = values[row_ids]
        vals = bs6n(z, knots, degree, basis - 1, n - basis - 1, h, far_h, deriv_count)
        if odd and basis <= order // 2 - 1:
            vals -= bs6n(
                z, -knots, degree, basis - 1, n - basis - 1, h, far_h, deriv_count
            )
        elif not odd and basis == 0:
            vals *= 2.0
        elif not odd and basis <= order // 2 - 1:
            vals += bs6n(
                z, -knots, degree, basis - 1, n - basis - 1, h, far_h, deriv_count
            )

        col = basis - 1 if odd else basis
        mats[:, row_ids, col] += vals

    row_ids = support_indices(values, mesh[n - 5], xx[-1])
    if row_ids.size:
        z = values[row_ids]
        last2 = xx[-order - 1 : -1]
        last = xx[-order:]
        a1 = bs6n(z, last2, degree, n, 0, h, far_h, deriv_count)
        a2 = bs6n(z, last, degree, n, 0, h, far_h, deriv_count)
        for basis in range(n - order, n):
            coeff_col = n - basis - 1
            vals = a1 * coeff[1, coeff_col] + a2 * coeff[0, coeff_col]
            col = basis - 1 if odd else basis
            mats[:, row_ids, col] += vals

    if values[0] == 0.0:
        if odd:
            for basis in range(1, order // 2):
                knots = xx[basis : basis + order]
                vals = bs6n(0.0, knots, degree, basis - 1, n - 1, h, far_h, deriv_count)
                vals -= bs6n(
                    0.0, -knots, degree, basis - 1, n - 1, h, far_h, deriv_count
                )
                mats[:, 0, basis - 1] += vals[:, 0]
        else:
            for basis in range(0, order // 2):
                knots = xx[basis : basis + order]
                vals = bs6n(0.0, knots, degree, basis - 1, n - 1, h, far_h, deriv_count)
                if basis == 0:
                    vals *= 2.0
                else:
                    vals += bs6n(
                        0.0, -knots, degree, basis - 1, n - 1, h, far_h, deriv_count
                    )
                mats[:, 0, basis] += vals[:, 0]

    return [mats[deriv] for deriv in range(deriv_count)]


def y_bs6_matrices(mesh: np.ndarray, values: np.ndarray, deriv_count: int) -> list[np.ndarray]:
    """Build the y-direction BS6_interp matrices for constant weights."""

    order = 7
    degree = order - 1
    n = mesh.size
    rate = mesh[-1] / mesh[-2]
    xx = extended_knots(mesh, rate, order)
    coeff = bscoe(order)
    h = mesh[1] - mesh[0]
    far_h = mesh[-1] - mesh[-2]
    mats = np.zeros((deriv_count, values.size, n), dtype=float)

    for basis in range(n):
        knots = xx[basis : basis + order]
        row_ids = support_indices(values, knots[0], knots[-1])
        if row_ids.size == 0:
            continue
        vals = bs6n(
            values[row_ids],
            knots,
            degree,
            basis - 1,
            n - basis - 1,
            h,
            far_h,
            deriv_count,
        )
        mats[:, row_ids, basis] += vals

    left1 = -xx[1:8]
    left2 = -xx[2:9]
    row_ids = support_indices(values, min(left1.min(), left2.min()), max(left1.max(), left2.max()))
    if row_ids.size:
        z = values[row_ids]
        a1 = bs6n(z, left1, degree, 0, n, h, far_h, deriv_count)
        a2 = bs6n(z, left2, degree, 0, n, h, far_h, deriv_count)
        for basis in range(order):
            vals = a1 * coeff[1, basis] + a2 * coeff[0, basis]
            mats[:, row_ids, basis] += vals

    row_ids = support_indices(values, mesh[n - 5], xx[-1])
    if row_ids.size:
        z = values[row_ids]
        last2 = xx[-order - 1 : -1]
        last = xx[-order:]
        a1 = bs6n(z, last2, degree, n, 0, h, far_h, deriv_count)
        a2 = bs6n(z, last, degree, n, 0, h, far_h, deriv_count)
        for basis in range(n - order, n):
            coeff_col = n - basis - 1
            vals = a1 * coeff[1, coeff_col] + a2 * coeff[0, coeff_col]
            mats[:, row_ids, basis] += vals

    return [mats[deriv] for deriv in range(deriv_count)]


def bs6mat(
    support_mesh: np.ndarray, value_mesh: np.ndarray, deriv_count: int, parity: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (
        x_bs6_matrices(support_mesh, value_mesh, deriv_count, parity),
        y_bs6_matrices(support_mesh, value_mesh, deriv_count),
    )


def svd_order(u: np.ndarray, singular_values: np.ndarray, v: np.ndarray, eps: float) -> int:
    for index, sigma in enumerate(singular_values):
        contribution = rms(u[:, index]) * sigma * rms(v[:, index])
        if contribution <= eps:
            return index + 1
    return singular_values.size


def uv_remesh(
    ucoef: np.ndarray,
    vcoef: np.ndarray,
    old_mesh: np.ndarray,
    new_mesh: np.ndarray,
    parity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse_x, coarse_y = bs6mat(old_mesh, new_mesh, 1, parity)
    ufit = coarse_x[0] @ ucoef
    vfit = coarse_y[0] @ vcoef

    remesh_x, remesh_y = bs6mat(new_mesh, new_mesh, 1, parity)
    if parity == 1:
        ucoef = np.linalg.solve(remesh_x[0][1:], ufit[1:])
    else:
        ucoef = np.linalg.solve(remesh_x[0], ufit)
    vcoef = np.linalg.solve(remesh_y[0], vfit)

    fine_x, fine_y = bs6mat(new_mesh, old_mesh, 2, parity)
    return (
        fine_x[0] @ ucoef,
        fine_y[0] @ vcoef,
        fine_x[1] @ ucoef,
        fine_y[1] @ vcoef,
    )


def svd_fit(
    field: np.ndarray, x1: np.ndarray, x2: np.ndarray, eps_svd: float, eps_fit: float, parity: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, singular_values, vh = np.linalg.svd(field, full_matrices=True)
    v = vh.T
    keep = svd_order(u, singular_values, v, eps_svd)
    u = u[:, :keep]
    singular_values = singular_values[:keep]
    v = v[:, :keep]
    if parity == 1:
        u[0, :] = 0.0

    ufit = u.copy()
    vfit = v.copy()
    dufit = np.zeros_like(u)
    dvfit = np.zeros_like(v)

    basis_x, basis_y = bs6mat(x1, x2, 1, parity)
    if parity == 1:
        ucoef = np.linalg.solve(basis_x[0][1:], u[1:])
    else:
        ucoef = np.linalg.solve(basis_x[0], u)
    vcoef = np.linalg.solve(basis_y[0], v)

    for step in range(1, 11):
        mesh_ids = np.concatenate((np.arange(0, 480, step), np.arange(480, 720)))
        mesh = x1[mesh_ids]
        umesh, vmesh, dumesh, dvmesh = uv_remesh(ucoef, vcoef, x1, mesh, parity)
        uerr = np.sqrt(np.mean((u - umesh) ** 2, axis=0)) * singular_values
        verr = np.sqrt(np.mean((v - vmesh) ** 2, axis=0)) * singular_values
        uok = uerr < eps_fit
        vok = verr < eps_fit
        ufit[:, uok] = umesh[:, uok]
        vfit[:, vok] = vmesh[:, vok]
        dufit[:, uok] = dumesh[:, uok]
        dvfit[:, vok] = dvmesh[:, vok]

    weighted_u = ufit * singular_values[None, :]
    weighted_du = dufit * singular_values[None, :]
    return (
        weighted_u @ vfit.T,
        weighted_du @ vfit.T,
        weighted_u @ dvfit.T,
    )


def fit_profile_scaled(
    field: np.ndarray, x1: np.ndarray, x2: np.ndarray, eps_svd: float, eps_fit: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1_col = x1[:, None]
    x2_row = x2[None, :]
    factor = np.sqrt(1.0 + x1_col**2) / np.sqrt(1.0 + x1_col**2 + x2_row**2)
    scaled_fit, scaled_x1, scaled_x2 = svd_fit(field / factor, x1, x2, eps_svd, eps_fit)
    fit = scaled_fit * factor
    fit_x1 = factor * (
        scaled_x1
        + scaled_fit
        * x1_col
        * x2_row**2
        / (1.0 + x1_col**2)
        / (1.0 + x1_col**2 + x2_row**2)
    )
    fit_x2 = factor * (
        scaled_x2 - scaled_fit * x2_row / (1.0 + x1_col**2 + x2_row**2)
    )
    return fit, fit_x1, fit_x2


def fit_u1_damped(
    u1: np.ndarray, x1: np.ndarray, x2: np.ndarray, eps_svd: float, eps_fit: float, power: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor0 = 1.0 + x1[:, None] ** 2
    factor = factor0 ** (power / 2.0)
    fit, fit_x1, fit_x2 = svd_fit(u1 / factor, x1, x2, eps_svd, eps_fit)
    fit[0, :] = 0.0
    fit_x2[0, :] = 0.0
    return (
        fit * factor,
        fit_x1 * factor + power * x1[:, None] * factor * fit / factor0,
        fit_x2 * factor,
    )


def fit_u2_damped(
    u2: np.ndarray, x1: np.ndarray, x2: np.ndarray, eps_svd: float, eps_fit: float, power: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor0 = 1.0 + x2[None, :] ** 2
    factor = factor0 ** (power / 2.0)
    fit, fit_x1, fit_x2 = svd_fit(u2 / factor, x1, x2, eps_svd, eps_fit, parity=0)
    fit[:, 0] = 0.0
    fit_x1[:, 0] = 0.0
    return (
        fit * factor,
        fit_x1 * factor,
        fit_x2 * factor + power * x2[None, :] * factor * fit / factor0,
    )


class Jet:
    """Truncated Taylor coefficients for vectorized one-variable AD."""

    def __init__(self, coeff: np.ndarray):
        self.coeff = coeff

    @property
    def order(self) -> int:
        return self.coeff.shape[0] - 1

    @classmethod
    def constant(cls, value: np.ndarray | float, order: int, template: np.ndarray) -> "Jet":
        coeff = np.zeros((order + 1, template.size), dtype=float)
        coeff[0] = value
        return cls(coeff)

    def __add__(self, other: "Jet") -> "Jet":
        return Jet(self.coeff + other.coeff)

    def __sub__(self, other: "Jet") -> "Jet":
        return Jet(self.coeff - other.coeff)

    def __neg__(self) -> "Jet":
        return Jet(-self.coeff)

    def __mul__(self, other: "Jet" | float) -> "Jet":
        if np.isscalar(other):
            return Jet(self.coeff * float(other))
        out = np.zeros_like(self.coeff)
        for degree in range(self.order + 1):
            for left_degree in range(degree + 1):
                out[degree] += self.coeff[left_degree] * other.coeff[degree - left_degree]
        return Jet(out)

    def __rmul__(self, other: float) -> "Jet":
        return self * other

    def derivative(self) -> "Jet":
        out = np.zeros_like(self.coeff)
        for degree in range(self.order):
            out[degree] = (degree + 1) * self.coeff[degree + 1]
        return Jet(out)

    @property
    def value(self) -> np.ndarray:
        return self.coeff[0]


def trig_jet(beta: np.ndarray, order: int, kind: str) -> Jet:
    coeff = np.empty((order + 1, beta.size), dtype=float)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    sin_cycle = (sin_beta, cos_beta, -sin_beta, -cos_beta)
    cos_cycle = (cos_beta, -sin_beta, -cos_beta, sin_beta)
    cycle = sin_cycle if kind == "sin" else cos_cycle
    for degree in range(order + 1):
        coeff[degree] = cycle[degree % 4] / math.factorial(degree)
    return Jet(coeff)


def ag_jets(beta: np.ndarray, order: int) -> dict[tuple[int, int, int, int], Jet]:
    """Build AG coefficient functions by the recurrence in Build_deri_r_ag.m."""

    zero = Jet.constant(0.0, order, beta)
    one = Jet.constant(1.0, order, beta)
    sin_beta = trig_jet(beta, order, "sin")
    cos_beta = trig_jet(beta, order, "cos")
    ag: dict[tuple[int, int, int, int], Jet] = {(0, 0, 0, 0): one}

    def get(i: int, j: int, k: int, ell: int) -> Jet:
        return ag.get((i, j, k, ell), zero)

    for degree in range(1, order + 1):
        for i in range(degree + 1):
            j = degree - i
            for k in range(degree + 1):
                for ell in range(degree - k + 1):
                    total = zero
                    if j >= 1:
                        if k + ell <= degree - 1:
                            previous = get(i, j - 1, k, ell)
                            total = total + cos_beta * previous.derivative()
                            total = total + (k - i - (j - 1)) * sin_beta * previous
                        if k >= 1:
                            total = total + sin_beta * get(i, j - 1, k - 1, ell)
                        if ell >= 1:
                            total = total + cos_beta * get(i, j - 1, k, ell - 1)
                    else:
                        if k + ell <= degree - 1:
                            previous = get(i - 1, j, k, ell)
                            total = total - sin_beta * previous.derivative()
                            total = total + (k - i - j + 1) * cos_beta * previous
                        if k >= 1:
                            total = total + cos_beta * get(i - 1, j, k - 1, ell)
                        if ell >= 1:
                            total = total - sin_beta * get(i - 1, j, k, ell - 1)
                    ag[(i, j, k, ell)] = total
    return ag


def polynomial_derivative_coeffs(deriv: int) -> np.ndarray:
    """Polynomial P where d^n[t^7/(1+t^2)^(7/2)] = P/(1+t^2)^(7/2+n)."""

    coeff = np.zeros(8, dtype=float)
    coeff[7] = 1.0
    for current in range(deriv):
        diff = np.arange(1, coeff.size, dtype=float) * coeff[1:]
        one_plus_t2_diff = np.zeros(diff.size + 2, dtype=float)
        one_plus_t2_diff[: diff.size] += diff
        one_plus_t2_diff[2:] += diff
        t_coeff = np.pad(coeff, (1, 0))
        size = max(one_plus_t2_diff.size, t_coeff.size)
        coeff = np.pad(one_plus_t2_diff, (0, size - one_plus_t2_diff.size))
        coeff -= (7 + 2 * current) * np.pad(t_coeff, (0, size - t_coeff.size))
    return coeff


def chi10_derivative(x: np.ndarray, deriv: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    mask = x >= 0.0
    z = x[mask]
    coeff = polynomial_derivative_coeffs(deriv)
    out[mask] = np.polynomial.polynomial.polyval(z, coeff) / (1.0 + z**2) ** (3.5 + deriv)
    return out


def reciprocal_series(coeff: np.ndarray) -> np.ndarray:
    out = np.zeros_like(coeff)
    out[0] = 1.0 / coeff[0]
    for degree in range(1, coeff.shape[0]):
        for inner in range(1, degree + 1):
            out[degree] -= coeff[inner] * out[degree - inner]
        out[degree] /= coeff[0]
    return out


def exp_series(coeff: np.ndarray) -> np.ndarray:
    out = np.zeros_like(coeff)
    out[0] = np.exp(coeff[0])
    for degree in range(1, coeff.shape[0]):
        for inner in range(1, degree + 1):
            out[degree] += inner * coeff[inner] * out[degree - inner]
        out[degree] /= degree
    return out


def chi0_raw_derivative(x: np.ndarray, deriv: int) -> np.ndarray:
    p = np.empty((deriv + 1, x.size), dtype=float)
    for degree in range(deriv + 1):
        p[degree] = (-1) ** degree * (x ** (-degree - 1) + (x - 1.0) ** (-degree - 1))
    denom = exp_series(p)
    denom[0] += 1.0
    cutoff = reciprocal_series(denom)
    return cutoff[deriv] * math.factorial(deriv)


def chi20_derivative(x: np.ndarray, deriv: int) -> np.ndarray:
    """Exponential cutoff derivative represented by MATLAB Chi20 handles."""

    out = np.zeros_like(x, dtype=float)
    upper = (x > 0.5) & (x < 1.0)
    lower = (x > 0.0) & (x <= 0.5)
    if np.any(upper):
        out[upper] = chi0_raw_derivative(x[upper], deriv)
    if np.any(lower):
        mirrored = chi0_raw_derivative(1.0 - x[lower], deriv)
        if deriv == 0:
            out[lower] = 1.0 - mirrored
        else:
            out[lower] = (-1) ** (deriv + 1) * mirrored
    if deriv == 0:
        out[x >= 1.0] = 1.0
    return out


def chi_derivatives(
    radius: np.ndarray, max_deriv: int, a1: float, lam1: float, a2: float
) -> list[np.ndarray]:
    scaled1 = (radius - a1) / math.sqrt(lam1)
    scaled2 = (radius - a2) / (9.0 * a2)
    chi1 = [
        chi10_derivative(scaled1, deriv) * lam1 ** (-deriv / 2.0)
        for deriv in range(max_deriv + 1)
    ]
    chi2 = [
        chi20_derivative(scaled2, deriv) * (9.0 * a2) ** (-deriv)
        for deriv in range(max_deriv + 1)
    ]

    assembled: list[np.ndarray] = []
    for deriv in range(max_deriv + 1):
        value = chi1[deriv].copy()
        for split in range(deriv + 1):
            remainder = deriv - split
            if split == deriv:
                factor = 1.0 - chi1[remainder]
            else:
                factor = -chi1[remainder]
            value += math.comb(deriv, split) * chi2[split] * factor
        assembled.append(value)
    return assembled


def psi_radial_derivatives(radius: np.ndarray, alpha: float, max_deriv: int) -> list[np.ndarray]:
    chi = chi_derivatives(radius, max_deriv, a1=10.0, lam1=50000.0, a2=100000.0)
    # Polar coefficients only consume radii above the cutoff r > 10. Avoid a
    # removable r = 0 negative-power warning while building the full grid array.
    safe_radius = np.where(radius == 0.0, 1.0, radius)
    powers: list[np.ndarray] = []
    factor = 1.0
    for deriv in range(max_deriv + 1):
        if deriv:
            factor *= 2.0 - alpha - deriv + 1.0
        powers.append(factor * safe_radius ** (2.0 - alpha - deriv))

    product_derivs: list[np.ndarray] = []
    for deriv in range(max_deriv + 1):
        value = np.zeros_like(radius, dtype=float)
        for split in range(deriv + 1):
            value += math.comb(deriv, split) * chi[split] * powers[deriv - split]
        product_derivs.append(value)
    return product_derivs


def deri_polar_agcoe(
    radius: np.ndarray, beta: np.ndarray, radial_derivs: list[np.ndarray], order: int, cutoff: float
) -> dict[tuple[int, int, int], np.ndarray]:
    selected = radius > cutoff
    ag = ag_jets(beta[selected], order)
    coe: dict[tuple[int, int, int], np.ndarray] = {}
    for degree in range(order + 1):
        for i in range(degree + 1):
            j = degree - i
            for ell in range(degree + 1):
                value = np.zeros_like(radius, dtype=float)
                selected_value = np.zeros(np.count_nonzero(selected), dtype=float)
                for k in range(degree - ell + 1):
                    selected_value += (
                        ag[(i, j, k, ell)].value
                        * radial_derivs[k][selected]
                        / radius[selected] ** (degree - k)
                    )
                value[selected] = selected_value
                coe[(i, j, ell)] = value
    return coe


def xycoef(x1: np.ndarray, x2: np.ndarray, alpha: float, order: int) -> dict[tuple[int, int, int], np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        radius_grid = np.sqrt(x1[:, None] ** 2 + x2[None, :] ** 2)
        beta_grid = np.arctan(x2[None, :] / x1[:, None])
    radius = radius_grid.reshape(-1, order="F")
    beta = beta_grid.reshape(-1, order="F")

    first_far = int(np.flatnonzero(x1 > 9.0)[0]) - 1
    h = x1[-1] - x1[-2]
    tail_count = int(np.rint((math.sqrt(x1[-1] ** 2 + x2[-1] ** 2) - x1[-1]) / h)) + 3
    radial_mesh = np.concatenate((x1[first_far:], x1[-1] + h * np.arange(1, tail_count + 1)))
    # The MATLAB call constructs two extra cutoff derivatives, but the polar
    # recurrence used here needs derivative orders 0..order.
    del radial_mesh
    radial_derivs = psi_radial_derivatives(radius, alpha, order)
    return deri_polar_agcoe(radius, beta, radial_derivs, order, cutoff=10.0)


def deri_psi1(
    n1: int,
    n2: int,
    angular_coeff: np.ndarray,
    bs_wg: np.ndarray,
    polar_coeff: dict[tuple[int, int, int], np.ndarray],
    order: int,
) -> dict[tuple[int, int], np.ndarray]:
    angle_derivs = [
        (bs_wg[deriv] @ angular_coeff) * (-1) ** deriv for deriv in range(order + 1)
    ]
    psi: dict[tuple[int, int], np.ndarray] = {}
    for degree in range(order + 1):
        for i in range(degree + 1):
            j = degree - i
            value = np.zeros(n1 * n2, dtype=float)
            for ell in range(degree + 1):
                value += polar_coeff[(i, j, ell)] * angle_derivs[ell]
            psi[(i, j)] = value.reshape((n1, n2), order="F")
    return psi


def omega_residual(
    cl: float,
    cw: float,
    x1: np.ndarray,
    x2: np.ndarray,
    omega: np.ndarray,
    zeta: np.ndarray,
    omega_x1: np.ndarray,
    omega_x2: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    zeta_x1: np.ndarray,
) -> np.ndarray:
    theta_x = zeta + x1[:, None] * zeta_x1
    return (
        -(cl * x1[:, None] + u1) * omega_x1
        - (cl * x2[None, :] + u2) * omega_x2
        + cw * omega
        + theta_x
    )


def zeta_residual(
    cl: float,
    cw: float,
    x1: np.ndarray,
    x2: np.ndarray,
    zeta: np.ndarray,
    zeta_x1: np.ndarray,
    zeta_x2: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    u1dx1 = np.zeros_like(u1)
    u1dx1[1:] = u1[1:] / x1[1:, None]
    return (
        -(cl * x1[:, None] + u1) * zeta_x1
        - (cl * x2[None, :] + u2) * zeta_x2
        + (2.0 * cw - u1dx1) * zeta
    )


def run(data_path: Path) -> dict[str, np.ndarray | float]:
    data = load_numeric_data(data_path)
    x1 = np.asarray(data["x1"], dtype=float)
    x2 = np.asarray(data["x2"], dtype=float)
    omega = np.asarray(data["w"], dtype=float)
    zeta = np.asarray(data["v"], dtype=float)
    omega_x1 = np.asarray(data["wx1"], dtype=float)
    omega_x2 = np.asarray(data["wx2"], dtype=float)
    zeta_x1 = np.asarray(data["vx1"], dtype=float)
    zeta_x2 = np.asarray(data["vx2"], dtype=float)
    vel = data["Vel"]

    omegafit, omegafit_x1, omegafit_x2 = fit_profile_scaled(
        omega, x1, x2, 1.0e-10, 1.0e-10
    )
    zetafit, zetafit_x1, zetafit_x2 = fit_profile_scaled(
        zeta, x1, x2, 1.0e-10, 1.0e-10
    )
    omega_checks = np.array(
        (rms(omega - omegafit), rms(omega_x1 - omegafit_x1), rms(omega_x2 - omegafit_x2))
    )
    zeta_checks = np.array(
        (rms(zeta - zetafit), rms(zeta_x1 - zetafit_x1), rms(zeta_x2 - zetafit_x2))
    )

    u1 = np.asarray(vel["u1"], dtype=float)
    u2 = np.asarray(vel["u2"], dtype=float)
    u10f = np.asarray(vel["u10f"], dtype=float)
    u20f = np.asarray(vel["u20f"], dtype=float)
    rat = float(np.asarray(data["rec"])[6])

    u10ffit, u10fx1, u10fx2 = fit_u1_damped(u10f, x1, x2, 1.0e-11, 1.0e-11, 0.5)
    polar_coeff = xycoef(
        np.asarray(data["gx1"], dtype=float),
        np.asarray(data["gx2"], dtype=float),
        float(data["alpha_b"]),
        order=3,
    )
    psi1 = deri_psi1(
        len(data["gx1"]),
        len(data["gx2"]),
        np.asarray(data["p_ag_coe"], dtype=float),
        np.asarray(data["BS1d_large"], dtype=object),
        polar_coeff,
        order=2,
    )
    u20ffit, u20fx1, u20fx2 = fit_u2_damped(u20f, x1, x2, 1.0e-11, 1.0e-11, 0.5)

    n1 = x1.size
    n2 = x2.size
    u1fit = u10ffit - rat * psi1[(0, 1)][:n1, :n2]
    u1x1 = u10fx1 - rat * psi1[(1, 1)][:n1, :n2]
    u1x2 = u10fx2 - rat * psi1[(0, 2)][:n1, :n2]
    u2fit = u20ffit + rat * psi1[(1, 0)][:n1, :n2]
    u2x1 = u20fx1 + rat * psi1[(2, 0)][:n1, :n2]
    u2x2 = u20fx2 + rat * psi1[(1, 1)][:n1, :n2]

    cl = float(4.0 * zeta_x1[0, 0] / omega_x1[0, 0])
    cw = float(vel["u1dx1"][0, 0] + cl / 2.0)
    fomega_original = omega_residual(
        cl, cw, x1, x2, omega, zeta, omega_x1, omega_x2, u1, u2, zeta_x1
    )
    fomega_fitted = omega_residual(
        cl, cw, x1, x2, omegafit, zetafit, omegafit_x1, omegafit_x2, u1fit, u2fit, zetafit_x1
    )
    fzeta_original = zeta_residual(cl, cw, x1, x2, zeta, zeta_x1, zeta_x2, u1, u2)
    fzeta_fitted = zeta_residual(cl, cw, x1, x2, zetafit, zetafit_x1, zetafit_x2, u1fit, u2fit)

    return {
        "omega_checks": omega_checks,
        "zeta_checks": zeta_checks,
        "u1_relative": rms(u1 - u1fit) / rms(u1),
        "u2_relative": rms(u2 - u2fit) / rms(u2),
        "divergence_rms": rms(u1x1 + u2x2),
        "vorticity_rms": rms(u1x2 - u2x1 - omegafit),
        "fomega_checks": np.array(
            (
                np.max(np.abs(fomega_original)),
                np.max(np.abs(fomega_fitted)),
                rms(fomega_original),
                rms(fomega_fitted),
            )
        ),
        "fzeta_checks": np.array(
            (
                np.max(np.abs(fzeta_original)),
                np.max(np.abs(fzeta_fitted)),
                rms(fzeta_original),
                rms(fzeta_fitted),
            )
        ),
    }


def print_results(result: dict[str, np.ndarray | float]) -> None:
    np.set_printoptions(precision=10, suppress=False)
    print("omega fit RMS checks: ", result["omega_checks"])
    print("zeta fit RMS checks:  ", result["zeta_checks"])
    print("u1 relative RMS fit error:      ", f"{result['u1_relative']:.10e}")
    print("u2 relative RMS fit error:      ", f"{result['u2_relative']:.10e}")
    print("RMS u1x1 + u2x2:                ", f"{result['divergence_rms']:.10e}")
    print("RMS u1x2 - u2x1 - omegafit:     ", f"{result['vorticity_rms']:.10e}")
    print("Fomega [max original, max fitted, RMS original, RMS fitted]:")
    print(result["fomega_checks"])
    print("Fzeta [max original, max fitted, RMS original, RMS fitted]:")
    print(result["fzeta_checks"])


if __name__ == "__main__":
    print_results(run(Path(__file__).with_name("data.mat")))
