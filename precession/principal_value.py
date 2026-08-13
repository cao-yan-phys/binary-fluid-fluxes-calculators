"""Pole-subtracted radial principal-value integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.integrate import quad_vec

from .conservative_kernels import ClassicalFluid, ConservativeMedium, QuantumFluid, radial_measure_prefactor


VectorFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class PVIntegralResult:
    """Measure-reduced conservative radial integral and its numerical estimate."""

    value: np.ndarray
    error: np.ndarray
    pole: float
    kappa: float | None
    used_subtraction: bool
    k_min: float
    k_max: float


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 4:
        raise ValueError("radial quadrature order must be at least 4")
    return np.polynomial.legendre.leggauss(order)


def _integrate_vector(function: VectorFunction, lower: float, upper: float, order: int) -> np.ndarray:
    if upper <= lower:
        return np.zeros_like(np.asarray(function(max(lower, 1.0e-30)), dtype=np.float64))
    nodes, weights = _legendre_rule(int(order))
    points = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    values = [np.asarray(function(float(point)), dtype=np.float64) for point in points]
    return 0.5 * (upper - lower) * np.tensordot(weights, np.stack(values), axes=(0, 0))


def _small_pole_intervals(lower: float, upper: float, k_pole: float) -> tuple[tuple[float, float], ...]:
    """Build geometric radial panels when the pole is far below ``upper``."""

    if not (lower == 0.0 and 0.0 < k_pole < upper / 64.0):
        return ((lower, upper),)
    edge = min(16.0 * k_pole, upper)
    intervals: list[tuple[float, float]] = [(lower, edge)]
    while edge < upper:
        next_edge = min(16.0 * edge, upper)
        intervals.append((edge, next_edge))
        edge = next_edge
    return tuple(intervals)


def _integrate_panels(function: VectorFunction, intervals: tuple[tuple[float, float], ...], order: int) -> np.ndarray:
    total: np.ndarray | None = None
    for lower, upper in intervals:
        value = _integrate_vector(function, lower, upper, order)
        total = value if total is None else total + value
    assert total is not None
    return total


def _prefetch_panels(
    function: VectorFunction,
    intervals: tuple[tuple[float, float], ...],
    order: int,
    *extra: float,
) -> None:
    """Prefetch all Gauss nodes in one CUDA source evaluation."""

    prefetch = getattr(function, "prefetch", None)
    if prefetch is None:
        return
    nodes, _ = _legendre_rule(int(order))
    points = [0.5 * (upper - lower) * nodes + 0.5 * (upper + lower) for lower, upper in intervals]
    if extra:
        points.append(np.asarray(extra, dtype=np.float64))
    prefetch(np.concatenate(points))


def _prefetch_quadrature(function: VectorFunction, lower: float, upper: float, order: int, *extra: float) -> None:
    """Give a CUDA-backed angular function all nodes needed by one PV pass."""

    prefetch = getattr(function, "prefetch", None)
    if prefetch is None:
        return
    nodes, _ = _legendre_rule(int(order))
    points = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    if extra:
        points = np.concatenate((points, np.asarray(extra, dtype=np.float64)))
    prefetch(points)


def _pv_log_remainder(k_pole: float, lower: float, upper: float) -> float:
    """PV integral of `1/(k_pole^2-k^2)` on `[lower, upper]`."""

    def primitive(value: float) -> float:
        return math.log(abs((k_pole + value) / (k_pole - value))) / (2.0 * k_pole)

    return primitive(upper) - primitive(lower)


def _has_interior_pole(k_pole: float, k_min: float, k_max: float) -> bool:
    scale = max(k_max, np.finfo(float).tiny)
    return k_pole > k_min + 1.0e-13 * scale and k_pole < k_max - 1.0e-13 * scale


def _classical_once(
    medium: ClassicalFluid,
    omega: float,
    q_function: VectorFunction,
    *,
    k_min: float,
    k_max: float,
    order: int,
) -> tuple[np.ndarray, bool, float, None]:
    k_pole, _ = medium.roots(omega)
    if not _has_interior_pole(k_pole, k_min, k_max):
        _prefetch_quadrature(q_function, k_min, k_max, order)
        value = _integrate_vector(
            lambda k: q_function(k) / medium.denominator(omega, k),
            k_min,
            k_max,
            order,
        )
        return value, False, k_pole, None

    _prefetch_quadrature(q_function, k_min, k_max, order, k_pole)
    q_pole = np.asarray(q_function(k_pole), dtype=np.float64)
    c_s2 = medium.c_s * medium.c_s
    value = _integrate_vector(
        lambda k: (q_function(k) - q_pole) / medium.denominator(omega, k),
        k_min,
        k_max,
        order,
    )
    value = value + q_pole * _pv_log_remainder(k_pole, k_min, k_max) / c_s2
    return value, True, k_pole, None


def _quantum_once(
    medium: QuantumFluid,
    omega: float,
    q_function: VectorFunction,
    *,
    k_min: float,
    k_max: float,
    order: int,
) -> tuple[np.ndarray, bool, float, float]:
    k_pole, kappa = medium.roots(omega)
    assert kappa is not None
    if not _has_interior_pole(k_pole, k_min, k_max):
        _prefetch_quadrature(q_function, k_min, k_max, order)
        value = _integrate_vector(
            lambda k: q_function(k) / medium.denominator(omega, k),
            k_min,
            k_max,
            order,
        )
        return value, False, k_pole, kappa

    intervals = _small_pole_intervals(k_min, k_max, k_pole)
    _prefetch_panels(q_function, intervals, order, k_pole)
    q_pole = np.asarray(q_function(k_pole), dtype=np.float64)
    coefficient = 4.0 * medium.m_phi**2 / (k_pole * k_pole + kappa * kappa)
    value = _integrate_panels(
        lambda k: coefficient
        * (
            (q_function(k) - q_pole) / (k_pole * k_pole - k * k)
            + q_function(k) / (k * k + kappa * kappa)
        ),
        intervals,
        order,
    )
    value = value + coefficient * q_pole * _pv_log_remainder(k_pole, k_min, k_max)
    return value, True, k_pole, kappa


def principal_value_integral(
    medium: ConservativeMedium,
    omega: float,
    q_function: VectorFunction,
    *,
    k_min: float = 0.0,
    k_max: float,
    radial_order: int = 32,
) -> PVIntegralResult:
    """Integrate the measure-reduced conservative kernel with a PV subtraction.

    The input is the angular function
    ``Q(k) = integral dOmega Re[K_n^* K_n,e]``.  The returned result includes
    the factor `(4*pi)^2*rho_bar/(2*pi)^3` from the radial measure, but does
    not contain the zero/nonzero frequency pairing weight.
    """

    if k_min < 0.0 or k_max <= k_min:
        raise ValueError("require 0 <= k_min < k_max")
    if radial_order < 8:
        raise ValueError("radial_order must be at least 8")
    if isinstance(medium, ClassicalFluid):
        raw, used_subtraction, pole, kappa = _classical_once(
            medium, omega, q_function, k_min=k_min, k_max=k_max, order=radial_order
        )
        refined, _, _, _ = _classical_once(
            medium, omega, q_function, k_min=k_min, k_max=k_max, order=2 * radial_order
        )
    elif isinstance(medium, QuantumFluid):
        raw, used_subtraction, pole, kappa = _quantum_once(
            medium, omega, q_function, k_min=k_min, k_max=k_max, order=radial_order
        )
        refined, _, _, _ = _quantum_once(
            medium, omega, q_function, k_min=k_min, k_max=k_max, order=2 * radial_order
        )
    else:
        raise TypeError("unsupported conservative medium")
    prefactor = radial_measure_prefactor(medium)
    return PVIntegralResult(
        value=np.asarray(refined * prefactor, dtype=np.float64),
        error=np.asarray(np.abs(refined - raw) * abs(prefactor), dtype=np.float64),
        pole=float(pole),
        kappa=None if kappa is None else float(kappa),
        used_subtraction=bool(used_subtraction),
        k_min=float(k_min),
        k_max=float(k_max),
    )


def symmetric_excision_reference(
    medium: ConservativeMedium,
    omega: float,
    q_function: VectorFunction,
    *,
    k_min: float,
    k_max: float,
    epsilon_fraction: float = 1.0e-5,
) -> np.ndarray:
    """Independent symmetric-excision PV reference for validation tests."""

    if not (0.0 < epsilon_fraction < 0.1):
        raise ValueError("epsilon_fraction must lie between 0 and 0.1")
    k_pole, _ = medium.roots(omega)
    integrand = lambda k: np.asarray(q_function(float(k)), dtype=np.float64) / medium.denominator(omega, float(k))
    if not _has_interior_pole(k_pole, k_min, k_max):
        value, _ = quad_vec(integrand, k_min, k_max, epsabs=1.0e-10, epsrel=1.0e-10)
        return np.asarray(value * radial_measure_prefactor(medium), dtype=np.float64)
    epsilon = epsilon_fraction * k_pole
    left, _ = quad_vec(integrand, k_min, k_pole - epsilon, epsabs=1.0e-10, epsrel=1.0e-10)
    right, _ = quad_vec(integrand, k_pole + epsilon, k_max, epsabs=1.0e-10, epsrel=1.0e-10)
    return np.asarray((left + right) * radial_measure_prefactor(medium), dtype=np.float64)
