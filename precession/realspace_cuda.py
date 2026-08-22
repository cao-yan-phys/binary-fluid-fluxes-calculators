"""CUDA reductions for point-source real-space classical corrections."""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float64

from .pair_orbit import PairOrbitGeometry


THREADS_PER_BLOCK = 256
FOUR_PI = 4.0 * math.pi


@cuda.jit
def _classical_self_gravity_correction_kernel(
    n: int,
    omega: float,
    rho_bar: float,
    c_s: float,
    separation: np.ndarray,
    separation_e: np.ndarray,
    delta: np.ndarray,
    pair_weights: np.ndarray,
    output: np.ndarray,
) -> None:
    index = cuda.grid(1)
    thread = cuda.threadIdx.x
    total_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)
    self_1_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)
    self_2_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)
    cross_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)

    total = 0.0
    self_1 = 0.0
    self_2 = 0.0
    cross = 0.0
    n_points = separation.shape[1]
    pair_index = index // n_points
    point_index = index - pair_index * n_points
    if pair_index < 4:
        radius = separation[pair_index, point_index]
        radius_e = separation_e[pair_index, point_index]
        a_acoustic = omega * omega
        a_self_gravity = a_acoustic + FOUR_PI * rho_bar
        if a_acoustic == 0.0:
            derivative_acoustic = 2.0 * math.pi * rho_bar / (c_s * c_s)
        else:
            wave_acoustic = math.sqrt(a_acoustic) / c_s
            z_acoustic = wave_acoustic * radius
            if abs(z_acoustic) < 1.0e-4:
                bracket_acoustic = 0.5 * z_acoustic * z_acoustic - z_acoustic**4 / 8.0 + z_acoustic**6 / 144.0
            else:
                bracket_acoustic = z_acoustic * math.sin(z_acoustic) - (1.0 - math.cos(z_acoustic))
            if radius > 0.0:
                quotient_acoustic = bracket_acoustic / (radius * radius)
            else:
                quotient_acoustic = 0.5 * wave_acoustic * wave_acoustic
            derivative_acoustic = FOUR_PI * rho_bar * quotient_acoustic / a_acoustic
        if a_self_gravity == 0.0:
            derivative_self_gravity = 0.0
        else:
            wave_self_gravity = math.sqrt(a_self_gravity) / c_s
            z_self_gravity = wave_self_gravity * radius
            if abs(z_self_gravity) < 1.0e-4:
                bracket_self_gravity = (
                    0.5 * z_self_gravity * z_self_gravity
                    - z_self_gravity**4 / 8.0
                    + z_self_gravity**6 / 144.0
                )
            else:
                bracket_self_gravity = z_self_gravity * math.sin(z_self_gravity) - (1.0 - math.cos(z_self_gravity))
            if radius > 0.0:
                quotient_self_gravity = bracket_self_gravity / (radius * radius)
            else:
                quotient_self_gravity = 0.5 * wave_self_gravity * wave_self_gravity
            derivative_self_gravity = FOUR_PI * rho_bar * quotient_self_gravity / a_self_gravity
        value = (
            pair_weights[pair_index]
            * math.cos(n * delta[point_index])
            * (derivative_self_gravity - derivative_acoustic)
            * radius_e
        )
        total = value
        if pair_index == 0:
            self_1 = value
        elif pair_index == 3:
            self_2 = value
        else:
            cross = value

    total_shared[thread] = total
    self_1_shared[thread] = self_1
    self_2_shared[thread] = self_2
    cross_shared[thread] = cross
    cuda.syncthreads()
    stride = THREADS_PER_BLOCK // 2
    while stride > 0:
        if thread < stride:
            total_shared[thread] += total_shared[thread + stride]
            self_1_shared[thread] += self_1_shared[thread + stride]
            self_2_shared[thread] += self_2_shared[thread + stride]
            cross_shared[thread] += cross_shared[thread + stride]
        cuda.syncthreads()
        stride //= 2
    if thread == 0:
        cuda.atomic.add(output, 0, total_shared[0])
        cuda.atomic.add(output, 1, self_1_shared[0])
        cuda.atomic.add(output, 2, self_2_shared[0])
        cuda.atomic.add(output, 3, cross_shared[0])


class CudaClassicalSelfGravityCorrection:
    """Cached GPU geometry for the classical self-gravity correction sum."""

    def __init__(self, geometry: PairOrbitGeometry) -> None:
        pairs = tuple(geometry.pairs[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1)))
        separation = np.ascontiguousarray(np.stack([pair.separation.ravel() for pair in pairs]), dtype=np.float64)
        separation_e = np.ascontiguousarray(np.stack([pair.separation_e.ravel() for pair in pairs]), dtype=np.float64)
        delta = np.ascontiguousarray(geometry.delta.ravel(), dtype=np.float64)
        f1, f2 = geometry.mass_fractions
        pair_weights = np.ascontiguousarray((f1 * f1, f1 * f2, f2 * f1, f2 * f2), dtype=np.float64)
        self._d_separation = cuda.to_device(separation)
        self._d_separation_e = cuda.to_device(separation_e)
        self._d_delta = cuda.to_device(delta)
        self._d_pair_weights = cuda.to_device(pair_weights)
        self._d_output = cuda.to_device(np.zeros(4, dtype=np.float64))
        self._n_points = delta.size

    def correction_parts(self, n: int, omega: float, rho_bar: float, c_s: float) -> np.ndarray:
        """Return the double-average correction before its frequency weight."""

        self._d_output.copy_to_device(np.zeros(4, dtype=np.float64))
        work_items = 4 * self._n_points
        blocks = (work_items + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _classical_self_gravity_correction_kernel[blocks, THREADS_PER_BLOCK](
            int(n),
            float(omega),
            float(rho_bar),
            float(c_s),
            self._d_separation,
            self._d_separation_e,
            self._d_delta,
            self._d_pair_weights,
            self._d_output,
        )
        cuda.synchronize()
        return self._d_output.copy_to_host() / self._n_points


@cuda.jit
def _quantum_pair_average_kernel(
    n: int,
    omega: float,
    rho_bar: float,
    m_phi: float,
    c_s_squared: float,
    include_self_gravity: bool,
    static_mode: int,
    separation: np.ndarray,
    separation_e: np.ndarray,
    delta: np.ndarray,
    pair_weights: np.ndarray,
    output: np.ndarray,
) -> None:
    index = cuda.grid(1)
    thread = cuda.threadIdx.x
    total_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)
    self_1_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)
    self_2_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)
    cross_shared = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float64)

    total = 0.0
    self_1 = 0.0
    self_2 = 0.0
    cross = 0.0
    n_points = separation.shape[1]
    pair_index = index // n_points
    point_index = index - pair_index * n_points
    if pair_index < 4:
        radius = separation[pair_index, point_index]
        radius_e = separation_e[pair_index, point_index]
        derivative = 0.0
        if static_mode == 0:
            a_value = omega * omega
            if include_self_gravity:
                a_value += FOUR_PI * rho_bar
            if a_value > 0.0:
                coefficient = c_s_squared / (4.0 * m_phi * m_phi)
                discriminant = math.sqrt(coefficient * coefficient + a_value / (m_phi * m_phi))
                wave_squared = 2.0 * m_phi * m_phi * (discriminant - coefficient)
                kappa_squared = 2.0 * m_phi * m_phi * (discriminant + coefficient)
                wave_number = math.sqrt(wave_squared)
                kappa = math.sqrt(kappa_squared)
                denominator = wave_squared + kappa_squared
                wave_weight = wave_squared / denominator
                kappa_weight = kappa_squared / denominator
                z = wave_number * radius
                u = kappa * radius
                numerator = 1.0 - kappa_weight * math.cos(z) - wave_weight * math.exp(-u)
                numerator_prime = (
                    kappa_weight * wave_number * math.sin(z)
                    + wave_weight * kappa * math.exp(-u)
                )
                if radius > 0.0:
                    direct = numerator_prime / radius - numerator / (radius * radius)
                else:
                    direct = 0.0
                if abs(z) > 1.0e-4 or abs(u) > 1.0e-4:
                    derivative = FOUR_PI * rho_bar * direct / a_value
                else:
                    derivative = FOUR_PI * rho_bar * wave_weight * kappa**3 * radius / (3.0 * a_value)
        elif static_mode == 1:
            coefficient = c_s_squared / (4.0 * m_phi * m_phi)
            lam = math.sqrt(c_s_squared)
            z = lam * radius
            if abs(z) < 1.0e-4:
                quotient = -0.5 * lam * lam + lam**3 * radius / 3.0 - lam**4 * radius * radius / 8.0
            else:
                quotient = (z * math.exp(-z) - (1.0 - math.exp(-z))) / (radius * radius)
            derivative = 2.0 * math.pi * rho_bar * (1.0 + 2.0 * quotient / (lam * lam)) / coefficient
        else:
            coefficient = c_s_squared / (4.0 * m_phi * m_phi)
            wave_number = math.sqrt(-c_s_squared)
            z = wave_number * radius
            if abs(z) < 1.0e-4:
                if radius > 0.0:
                    quotient = (
                        0.5 * wave_number * wave_number
                        - z**4 / (8.0 * radius * radius)
                        + z**6 / (144.0 * radius * radius)
                    )
                else:
                    quotient = 0.5 * wave_number * wave_number
            else:
                quotient = (z * math.sin(z) - (1.0 - math.cos(z))) / (radius * radius)
            derivative = FOUR_PI * rho_bar * (
                0.5 / coefficient + quotient / (wave_number * wave_number * (-coefficient))
            )
        value = pair_weights[pair_index] * math.cos(n * delta[point_index]) * derivative * radius_e
        total = value
        if pair_index == 0:
            self_1 = value
        elif pair_index == 3:
            self_2 = value
        else:
            cross = value

    total_shared[thread] = total
    self_1_shared[thread] = self_1
    self_2_shared[thread] = self_2
    cross_shared[thread] = cross
    cuda.syncthreads()
    stride = THREADS_PER_BLOCK // 2
    while stride > 0:
        if thread < stride:
            total_shared[thread] += total_shared[thread + stride]
            self_1_shared[thread] += self_1_shared[thread + stride]
            self_2_shared[thread] += self_2_shared[thread + stride]
            cross_shared[thread] += cross_shared[thread + stride]
        cuda.syncthreads()
        stride //= 2
    if thread == 0:
        cuda.atomic.add(output, 0, total_shared[0])
        cuda.atomic.add(output, 1, self_1_shared[0])
        cuda.atomic.add(output, 2, self_2_shared[0])
        cuda.atomic.add(output, 3, cross_shared[0])


class CudaQuantumPairAverage:
    """Cached GPU geometry for quantum-fluid real-space harmonic averages."""

    def __init__(self, geometry: PairOrbitGeometry) -> None:
        pairs = tuple(geometry.pairs[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1)))
        separation = np.ascontiguousarray(np.stack([pair.separation.ravel() for pair in pairs]), dtype=np.float64)
        separation_e = np.ascontiguousarray(np.stack([pair.separation_e.ravel() for pair in pairs]), dtype=np.float64)
        delta = np.ascontiguousarray(geometry.delta.ravel(), dtype=np.float64)
        f1, f2 = geometry.mass_fractions
        pair_weights = np.ascontiguousarray((f1 * f1, f1 * f2, f2 * f1, f2 * f2), dtype=np.float64)
        self._d_separation = cuda.to_device(separation)
        self._d_separation_e = cuda.to_device(separation_e)
        self._d_delta = cuda.to_device(delta)
        self._d_pair_weights = cuda.to_device(pair_weights)
        self._d_output = cuda.to_device(np.zeros(4, dtype=np.float64))
        self._n_points = delta.size

    def pair_parts(
        self,
        n: int,
        omega: float,
        rho_bar: float,
        m_phi: float,
        c_s_squared: float,
        include_self_gravity: bool,
    ) -> np.ndarray:
        """Return the double average before its static or harmonic weight."""

        static_mode = 0
        if n == 0 and not include_self_gravity:
            static_mode = 1 if c_s_squared > 0.0 else 2
        self._d_output.copy_to_device(np.zeros(4, dtype=np.float64))
        work_items = 4 * self._n_points
        blocks = (work_items + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _quantum_pair_average_kernel[blocks, THREADS_PER_BLOCK](
            int(n),
            float(omega),
            float(rho_bar),
            float(m_phi),
            float(c_s_squared),
            bool(include_self_gravity),
            static_mode,
            self._d_separation,
            self._d_separation_e,
            self._d_delta,
            self._d_pair_weights,
            self._d_output,
        )
        cuda.synchronize()
        return self._d_output.copy_to_host() / self._n_points
