"""Kepler-orbit arrays and fixed-mean-anomaly eccentricity derivatives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class BinaryOrbit:
    """A prescribed Kepler binary in the medium rest frame.

    The public frequency is physical ``omega_tilde``.  When omitted, it is
    fixed by the Newtonian relation ``omega_tilde**2 = M/a**3``.
    """

    m1: float
    m2: float
    a: float
    e: float
    omega_tilde: float | None = None

    def __post_init__(self) -> None:
        if self.m1 <= 0.0 or self.m2 <= 0.0:
            raise ValueError("m1 and m2 must be positive")
        if self.a <= 0.0:
            raise ValueError("a must be positive")
        if not (0.0 <= self.e < 1.0):
            raise ValueError("e must satisfy 0 <= e < 1")
        implied = math.sqrt(self.total_mass / self.a**3)
        if self.omega_tilde is not None:
            if self.omega_tilde <= 0.0:
                raise ValueError("omega_tilde must be positive")
            if not math.isclose(self.omega_tilde, implied, rel_tol=1.0e-10, abs_tol=0.0):
                raise ValueError("omega_tilde must satisfy omega_tilde**2 = M/a**3")

    @property
    def total_mass(self) -> float:
        return self.m1 + self.m2

    @property
    def nu(self) -> float:
        return self.m1 * self.m2 / self.total_mass**2

    @property
    def mu(self) -> float:
        return self.nu * self.total_mass

    @property
    def f1(self) -> float:
        return self.m1 / self.total_mass

    @property
    def f2(self) -> float:
        return self.m2 / self.total_mass

    @property
    def physical_omega(self) -> float:
        if self.omega_tilde is not None:
            return self.omega_tilde
        return math.sqrt(self.total_mass / self.a**3)


@dataclass(frozen=True)
class KeplerGrid:
    """Cached quantities on a uniform mean-anomaly grid."""

    ell: np.ndarray
    xi: np.ndarray
    cos_xi: np.ndarray
    sin_xi: np.ndarray
    residual_max: float


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@lru_cache(maxsize=64)
def cached_kepler_grid(e: float, n_ell: int) -> KeplerGrid:
    """Solve Kepler's equation on a periodic, uniform mean-anomaly grid."""

    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if not _power_of_two(n_ell):
        raise ValueError("n_ell must be a positive power of two")

    ell = 2.0 * math.pi * np.arange(n_ell, dtype=np.float64) / n_ell
    if e == 0.0:
        xi = ell.copy()
    else:
        lower = ell - e
        upper = ell + e
        xi = ell + 0.85 * e * np.sign(np.sin(ell))
        xi = np.clip(xi, lower, upper)
        for _ in range(80):
            sin_xi = np.sin(xi)
            residual = xi - e * sin_xi - ell
            positive = residual > 0.0
            upper = np.where(positive, xi, upper)
            lower = np.where(positive, lower, xi)
            derivative = 1.0 - e * np.cos(xi)
            candidate = xi - residual / derivative
            outside = (candidate <= lower) | (candidate >= upper) | ~np.isfinite(candidate)
            candidate = np.where(outside, 0.5 * (lower + upper), candidate)
            if float(np.max(np.abs(candidate - xi))) < 2.0e-14:
                xi = candidate
                break
            xi = candidate
        else:
            raise RuntimeError("Kepler solver did not converge")

    residual_max = float(np.max(np.abs(xi - e * np.sin(xi) - ell)))
    if residual_max > 2.0e-13:
        raise RuntimeError(f"Kepler residual too large: {residual_max:.3e}")
    for array in (ell, xi):
        array.setflags(write=False)
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)
    cos_xi.setflags(write=False)
    sin_xi.setflags(write=False)
    return KeplerGrid(ell, xi, cos_xi, sin_xi, residual_max)


def relative_orbit_arrays(orbit: BinaryOrbit, n_ell: int) -> tuple[KeplerGrid, np.ndarray, np.ndarray]:
    """Return ``(grid, X, X_e)`` at fixed mean anomaly.

    ``X_e`` is ``partial_e X|_ell`` and is analytic, not a finite difference.
    """

    grid = cached_kepler_grid(float(orbit.e), int(n_ell))
    beta = math.sqrt(1.0 - orbit.e * orbit.e)
    jacobian = 1.0 - orbit.e * grid.cos_xi
    x = orbit.a * (grid.cos_xi - orbit.e)
    y = orbit.a * beta * grid.sin_xi
    zeros = np.zeros_like(x)
    position = np.stack((x, y, zeros), axis=-1)

    x_e = orbit.a * (-1.0 - grid.sin_xi**2 / jacobian)
    y_e = orbit.a * (
        -orbit.e * grid.sin_xi / beta
        + beta * grid.cos_xi * grid.sin_xi / jacobian
    )
    position_e = np.stack((x_e, y_e, zeros), axis=-1)
    return grid, position, position_e


def orbit_at_eccentric_anomaly(orbit: BinaryOrbit, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return relative position and mean anomaly for arbitrary ``xi`` values."""

    beta = math.sqrt(1.0 - orbit.e * orbit.e)
    position = np.stack(
        (
            orbit.a * (np.cos(xi) - orbit.e),
            orbit.a * beta * np.sin(xi),
            np.zeros_like(xi, dtype=np.float64),
        ),
        axis=-1,
    )
    return position, xi - orbit.e * np.sin(xi)
