"""Independent finite-resolution force reconstruction and Gauss projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .periastron_precession import PrecessionConfig, _angular_nodes, _default_k_max, _window
from .conservative_kernels import ConservativeMedium, static_ir_is_prescription_dependent
from .offshell_source import OffshellSource
from .orbit_derivatives import BinaryOrbit, orbit_at_eccentric_anomaly
from .principal_value import principal_value_integral


@dataclass(frozen=True)
class GaussBenchmarkResult:
    """Finite-resolution reconstructed-field Gauss-projection benchmark result."""

    delta_varpi_static: float | None
    delta_varpi_osc: float
    delta_varpi_total: float | None
    radial_error: float
    static_ir_status: str
    n_time: int


def gauss_precession_benchmark(
    orbit: BinaryOrbit,
    medium: ConservativeMedium,
    config: PrecessionConfig,
    *,
    n_time: int = 64,
) -> GaussBenchmarkResult:
    """Reconstruct a regulated real conservative field and apply Gauss' formula.

    This intentionally follows a different route from the Hamiltonian
    eccentricity derivative. It is intended for modest-resolution validation
    points and applies the same source window to both source and test factors.
    """

    if orbit.e <= 0.0:
        raise ValueError("Gauss periastron benchmark requires e > 0")
    if n_time < 16:
        raise ValueError("n_time must be at least 16")
    source = OffshellSource(orbit, n_ell=config.n_ell)
    directions, weights = _angular_nodes(config.n_mu, config.n_phi)
    k_max = _default_k_max(orbit, medium, config) if config.k_max is None else config.k_max
    assert k_max is not None
    xi = 2.0 * math.pi * np.arange(n_time, dtype=np.float64) / n_time
    relative_position, ell = orbit_at_eccentric_anomaly(orbit, xi)
    x1 = orbit.f2 * relative_position
    x2 = -orbit.f1 * relative_position
    mode_acceleration: dict[int, np.ndarray] = {}
    mode_error: dict[int, float] = {}
    static_ir = static_ir_is_prescription_dependent(medium)

    for n in range(config.n_max + 1):
        if n == 0 and static_ir:
            continue
        omega = n * orbit.physical_omega
        cache: dict[float, np.ndarray] = {}

        def force_mode(k: float) -> np.ndarray:
            key = float(k)
            if key in cache:
                return cache[key]
            result = np.zeros((n_time, 3), dtype=np.complex128)
            regulator = _window(key, config) ** 2
            for direction, weight in zip(directions, weights, strict=True):
                harmonic = source.on_shell_harmonic(n, key, direction)
                phases = np.exp(1j * key * (x1 @ direction)) - np.exp(1j * key * (x2 @ direction))
                result += weight * (-1j * orbit.total_mass * key * regulator) * phases[:, None] * harmonic.total * direction
            packed = np.concatenate((result.real.ravel(), result.imag.ravel()))
            cache[key] = packed
            return packed

        integral = principal_value_integral(
            medium,
            omega,
            force_mode,
            k_min=config.k_min,
            k_max=k_max,
            radial_order=config.radial_order,
        )
        packed = integral.value
        size = n_time * 3
        mode_acceleration[n] = packed[:size].reshape(n_time, 3) + 1j * packed[size:].reshape(n_time, 3)
        mode_error[n] = float(np.linalg.norm(integral.error))

    acceleration_static = np.zeros((n_time, 3), dtype=np.float64)
    if 0 in mode_acceleration:
        acceleration_static = mode_acceleration[0].real
    acceleration_osc = np.zeros((n_time, 3), dtype=np.float64)
    for n, value in mode_acceleration.items():
        if n > 0:
            acceleration_osc += 2.0 * np.real(np.exp(-1j * n * ell)[:, None] * value)

    beta = math.sqrt(1.0 - orbit.e * orbit.e)
    projection_x = beta * (np.cos(xi) ** 2 + orbit.e * np.cos(xi) - 2.0)
    projection_y = np.sin(xi) * (np.cos(xi) - orbit.e)

    def project(acceleration: np.ndarray) -> float:
        integrand = projection_x * acceleration[:, 0] + projection_y * acceleration[:, 1]
        return float(2.0 * math.pi * np.mean(integrand) / (orbit.a * orbit.physical_omega**2 * orbit.e))

    static_value = None if static_ir else project(acceleration_static)
    osc_value = project(acceleration_osc)
    total_value = None if static_ir else project(acceleration_static + acceleration_osc)
    return GaussBenchmarkResult(
        delta_varpi_static=static_value,
        delta_varpi_osc=osc_value,
        delta_varpi_total=total_value,
        radial_error=math.sqrt(sum(error * error for error in mode_error.values())),
        static_ir_status="PRESCRIPTION_DEPENDENT" if static_ir else "FINITE_UNDER_SELECTED_PRESCRIPTION",
        n_time=n_time,
    )
