"""Exact point-source real-space conservative response kernels."""

from __future__ import annotations

import math

import numpy as np

from .conservative_kernels import ClassicalFluid, QuantumFluid


FOUR_PI = 4.0 * math.pi


def _array_radius(radius: np.ndarray | float) -> np.ndarray:
    values = np.asarray(radius, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("radius must be non-negative")
    return values


def classical_g_n(fluid: ClassicalFluid, omega: float, radius: np.ndarray | float) -> np.ndarray:
    """Exact point-source kernel ``g_n(r)`` for a classical fluid."""

    r = _array_radius(radius)
    a_value = fluid.a_n(omega)
    if a_value == 0.0:
        if fluid.rho_bar == 0.0:
            return np.zeros_like(r)
        if fluid.include_self_gravity:
            raise ValueError("the kernel for a self-gravitating classical fluid has vanishing A_n")
        return 2.0 * math.pi * fluid.rho_bar * r / (fluid.c_s * fluid.c_s)
    wave_number = math.sqrt(a_value) / fluid.c_s
    z = wave_number * r
    numerator = np.where(
        np.abs(z) < 1.0e-4,
        0.5 * z * z - z**4 / 24.0 + z**6 / 720.0,
        1.0 - np.cos(z),
    )
    quotient = np.divide(numerator, r, out=np.zeros_like(r), where=r > 0.0)
    return FOUR_PI * fluid.rho_bar * quotient / a_value


def classical_gprime_n(fluid: ClassicalFluid, omega: float, radius: np.ndarray | float) -> np.ndarray:
    """Radial derivative of the exact point-source kernel for a classical fluid."""

    r = _array_radius(radius)
    a_value = fluid.a_n(omega)
    static_limit = 2.0 * math.pi * fluid.rho_bar / (fluid.c_s * fluid.c_s)
    if a_value == 0.0:
        if fluid.rho_bar == 0.0:
            return np.zeros_like(r)
        if fluid.include_self_gravity:
            raise ValueError("the kernel for a self-gravitating classical fluid has vanishing A_n")
        return np.full_like(r, static_limit, dtype=np.float64)
    wave_number = math.sqrt(a_value) / fluid.c_s
    z = wave_number * r
    bracket = np.where(
        np.abs(z) < 1.0e-4,
        0.5 * z * z - z**4 / 8.0 + z**6 / 144.0,
        z * np.sin(z) - (1.0 - np.cos(z)),
    )
    quotient = np.divide(bracket, r * r, out=np.full_like(r, 0.5 * wave_number * wave_number), where=r > 0.0)
    return FOUR_PI * fluid.rho_bar * quotient / a_value


def quantum_g_n(fluid: QuantumFluid, omega: float, radius: np.ndarray | float) -> np.ndarray:
    """Exact finite-frequency quantum-fluid point-source kernel."""

    r = _array_radius(radius)
    a_value = fluid.a_n(omega)
    if a_value <= 0.0:
        if a_value == 0.0 and fluid.rho_bar == 0.0:
            return np.zeros_like(r)
        raise ValueError("finite-frequency quantum-fluid kernel requires positive A_n; use a static special case")
    wave_number, kappa = fluid.roots(omega)
    assert kappa is not None
    denominator = wave_number * wave_number + kappa * kappa
    a_weight = wave_number * wave_number / denominator
    b_weight = kappa * kappa / denominator
    z = wave_number * r
    u = kappa * r
    numerator = 1.0 - b_weight * np.cos(z) - a_weight * np.exp(-u)
    zero_limit = a_weight * kappa
    quotient = np.divide(numerator, r, out=np.full_like(r, zero_limit), where=r > 0.0)
    small = np.maximum(np.abs(z), np.abs(u)) < 1.0e-4
    quotient = np.where(small, zero_limit + a_weight * kappa**3 * r * r / 6.0, quotient)
    return FOUR_PI * fluid.rho_bar * quotient / a_value


def quantum_gprime_n(fluid: QuantumFluid, omega: float, radius: np.ndarray | float) -> np.ndarray:
    """Radial derivative of the exact finite-frequency quantum-fluid kernel."""

    r = _array_radius(radius)
    a_value = fluid.a_n(omega)
    if a_value <= 0.0:
        if a_value == 0.0 and fluid.rho_bar == 0.0:
            return np.zeros_like(r)
        raise ValueError("quantum-fluid kernel requires positive A_n; use a static special case")
    wave_number, kappa = fluid.roots(omega)
    assert kappa is not None
    denominator = wave_number * wave_number + kappa * kappa
    a_weight = wave_number * wave_number / denominator
    b_weight = kappa * kappa / denominator
    z = wave_number * r
    u = kappa * r
    numerator = 1.0 - b_weight * np.cos(z) - a_weight * np.exp(-u)
    numerator_prime = b_weight * wave_number * np.sin(z) + a_weight * kappa * np.exp(-u)
    direct = np.divide(numerator_prime, r, out=np.zeros_like(r), where=r > 0.0) - np.divide(
        numerator, r * r, out=np.zeros_like(r), where=r > 0.0
    )
    small = np.maximum(np.abs(z), np.abs(u)) < 1.0e-4
    direct = np.where(small, a_weight * kappa**3 * r / 3.0, direct)
    return FOUR_PI * fluid.rho_bar * direct / a_value


def quantum_static_no_sg_finite_cs_g(fluid: QuantumFluid, radius: np.ndarray | float) -> np.ndarray:
    """Static quantum-fluid kernel without self-gravity and with positive ``c_S^2``."""

    c_S_squared = fluid.c_S_squared
    if fluid.include_self_gravity or c_S_squared <= 0.0:
        raise ValueError("finite-c_S static kernel requires a quantum fluid without self-gravity and c_S^2 > 0")
    r = _array_radius(radius)
    coefficient = fluid.physical_k2_coefficient
    lam = math.sqrt(c_S_squared)
    correction = np.divide(
        1.0 - np.exp(-lam * r),
        r,
        out=np.full_like(r, lam),
        where=r > 0.0,
    )
    return 2.0 * math.pi * fluid.rho_bar / coefficient * (r + 2.0 * correction / (lam * lam))


def quantum_static_no_sg_finite_cs_gprime(fluid: QuantumFluid, radius: np.ndarray | float) -> np.ndarray:
    """Analytic radial derivative of the finite-``c_S`` static quantum-fluid kernel."""

    c_S_squared = fluid.c_S_squared
    if fluid.include_self_gravity or c_S_squared <= 0.0:
        raise ValueError("finite-c_S static kernel requires a quantum fluid without self-gravity and c_S^2 > 0")
    r = _array_radius(radius)
    coefficient = fluid.physical_k2_coefficient
    lam = math.sqrt(c_S_squared)
    z = lam * r
    numerator = np.where(
        np.abs(z) < 1.0e-4,
        -0.5 * z * z + z**3 / 3.0 - z**4 / 8.0,
        z * np.exp(-z) - (1.0 - np.exp(-z)),
    )
    quotient = np.divide(numerator, r * r, out=np.full_like(r, -0.5 * lam * lam), where=r > 0.0)
    bracket = 1.0 + 2.0 * quotient / (lam * lam)
    return 2.0 * math.pi * fluid.rho_bar * bracket / coefficient


def quantum_static_no_sg_negative_cs2_g(fluid: QuantumFluid, radius: np.ndarray | float) -> np.ndarray:
    """Finite, orbit-dependent static kernel for negative ``c_S^2``.

    The ``A_0 -> 0`` kernel contains an additive infrared constant.  It
    drops out of the eccentricity derivative, so this function returns the
    uniquely relevant finite part.
    """

    c_S_squared = fluid.c_S_squared
    if fluid.include_self_gravity or c_S_squared >= 0.0:
        raise ValueError("negative-c_S^2 static kernel requires a quantum fluid without self-gravity and c_S^2 < 0")
    r = _array_radius(radius)
    coefficient = fluid.physical_k2_coefficient
    wave_number = math.sqrt(-c_S_squared)
    z = wave_number * r
    numerator = np.where(
        np.abs(z) < 1.0e-4,
        0.5 * z * z - z**4 / 24.0 + z**6 / 720.0,
        1.0 - np.cos(z),
    )
    quotient = np.divide(numerator, r, out=np.zeros_like(r), where=r > 0.0)
    return FOUR_PI * fluid.rho_bar * (
        0.5 * r / coefficient + quotient / (wave_number * wave_number * (-coefficient))
    )


def quantum_static_no_sg_negative_cs2_gprime(fluid: QuantumFluid, radius: np.ndarray | float) -> np.ndarray:
    """Derivative of the finite negative-``c_S^2`` static kernel."""

    c_S_squared = fluid.c_S_squared
    if fluid.include_self_gravity or c_S_squared >= 0.0:
        raise ValueError("negative-c_S^2 static kernel requires a quantum fluid without self-gravity and c_S^2 < 0")
    r = _array_radius(radius)
    coefficient = fluid.physical_k2_coefficient
    wave_number = math.sqrt(-c_S_squared)
    z = wave_number * r
    numerator = np.where(
        np.abs(z) < 1.0e-4,
        0.5 * z * z - z**4 / 8.0 + z**6 / 144.0,
        z * np.sin(z) - (1.0 - np.cos(z)),
    )
    quotient = np.divide(numerator, r * r, out=np.full_like(r, 0.5 * wave_number * wave_number), where=r > 0.0)
    return FOUR_PI * fluid.rho_bar * (
        0.5 / coefficient + quotient / (wave_number * wave_number * (-coefficient))
    )
