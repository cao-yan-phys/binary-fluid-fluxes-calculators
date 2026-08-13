"""Shared arbitrary-wave-vector source harmonics for conservative precession."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import cuda

from .orbit_derivatives import BinaryOrbit, relative_orbit_arrays


Backend = Literal["auto", "cuda", "cpu"]


def cuda_available() -> bool:
    """Return whether the Numba CUDA backend is usable."""

    try:
        return bool(cuda.is_available())
    except Exception:
        return False


@cuda.jit
def _quadratic_parts_cuda_kernel(
    k_values: np.ndarray,
    n: int,
    directions: np.ndarray,
    weights: np.ndarray,
    position: np.ndarray,
    position_e: np.ndarray,
    ell: np.ndarray,
    f1: float,
    f2: float,
    output: np.ndarray,
) -> None:
    """Accumulate weighted Re[K* K_e] source parts for one (n, k)."""

    index = cuda.grid(1)
    n_angles = directions.shape[0]
    k_index = index // n_angles
    angle_index = index - k_index * n_angles
    if k_index >= k_values.size:
        return

    k = k_values[k_index]
    dx = directions[angle_index, 0]
    dy = directions[angle_index, 1]
    dz = directions[angle_index, 2]
    k1_re = 0.0
    k1_im = 0.0
    k1e_re = 0.0
    k1e_im = 0.0
    k2_re = 0.0
    k2_im = 0.0
    k2e_re = 0.0
    k2e_im = 0.0
    for i_ell in range(ell.size):
        phase = k * (
            dx * position[i_ell, 0]
            + dy * position[i_ell, 1]
            + dz * position[i_ell, 2]
        )
        phase_e = k * (
            dx * position_e[i_ell, 0]
            + dy * position_e[i_ell, 1]
            + dz * position_e[i_ell, 2]
        )
        time_phase = n * ell[i_ell]
        time_re = math.cos(time_phase)
        time_im = math.sin(time_phase)

        angle_1 = f2 * phase
        body1_re = f1 * math.cos(angle_1)
        body1_im = -f1 * math.sin(angle_1)
        derivative_1 = f2 * phase_e
        body1e_re = body1_im * derivative_1
        body1e_im = -body1_re * derivative_1

        angle_2 = f1 * phase
        body2_re = f2 * math.cos(angle_2)
        body2_im = f2 * math.sin(angle_2)
        derivative_2 = f1 * phase_e
        body2e_re = -body2_im * derivative_2
        body2e_im = body2_re * derivative_2

        k1_re += time_re * body1_re - time_im * body1_im
        k1_im += time_re * body1_im + time_im * body1_re
        k1e_re += time_re * body1e_re - time_im * body1e_im
        k1e_im += time_re * body1e_im + time_im * body1e_re
        k2_re += time_re * body2_re - time_im * body2_im
        k2_im += time_re * body2_im + time_im * body2_re
        k2e_re += time_re * body2e_re - time_im * body2e_im
        k2e_im += time_re * body2e_im + time_im * body2e_re

    inverse_n_ell = 1.0 / ell.size
    k1_re *= inverse_n_ell
    k1_im *= inverse_n_ell
    k1e_re *= inverse_n_ell
    k1e_im *= inverse_n_ell
    k2_re *= inverse_n_ell
    k2_im *= inverse_n_ell
    k2e_re *= inverse_n_ell
    k2e_im *= inverse_n_ell

    total_re = k1_re + k2_re
    total_im = k1_im + k2_im
    total_e_re = k1e_re + k2e_re
    total_e_im = k1e_im + k2e_im
    weight = weights[angle_index]
    self_1 = weight * (k1_re * k1e_re + k1_im * k1e_im)
    self_2 = weight * (k2_re * k2e_re + k2_im * k2e_im)
    total = weight * (total_re * total_e_re + total_im * total_e_im)
    cuda.atomic.add(output, (k_index, 0), total)
    cuda.atomic.add(output, (k_index, 1), self_1)
    cuda.atomic.add(output, (k_index, 2), self_2)
    cuda.atomic.add(output, (k_index, 3), total - self_1 - self_2)


class CudaQuadraticSource:
    """GPU evaluator for angular source quadratic forms at arbitrary radial k."""

    def __init__(self, source: "OffshellSource", directions: np.ndarray, weights: np.ndarray) -> None:
        if not cuda_available():
            raise RuntimeError("CUDA backend requested, but numba.cuda is unavailable")
        self._source = source
        self._directions = np.ascontiguousarray(directions, dtype=np.float64)
        self._weights = np.ascontiguousarray(weights, dtype=np.float64)
        self._position = np.ascontiguousarray(source.position, dtype=np.float64)
        self._position_e = np.ascontiguousarray(source.position_e, dtype=np.float64)
        self._ell = np.ascontiguousarray(source.grid.ell, dtype=np.float64)
        self._d_directions = cuda.to_device(self._directions)
        self._d_weights = cuda.to_device(self._weights)
        self._d_position = cuda.to_device(self._position)
        self._d_position_e = cuda.to_device(self._position_e)
        self._d_ell = cuda.to_device(self._ell)
        self._threads_per_block = 128

    def quadratic_parts(self, n: int, k: float) -> np.ndarray:
        """Return angular integral of total/self-1/self-2/cross source parts."""

        return self.quadratic_parts_many(n, np.array((k,), dtype=np.float64))[0]

    def quadratic_parts_many(self, n: int, k_values: np.ndarray) -> np.ndarray:
        """Evaluate a batch of radial wave numbers in one CUDA launch."""

        k_values = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
        if k_values.ndim != 1:
            raise ValueError("k_values must be one-dimensional")
        if k_values.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        d_k_values = cuda.to_device(k_values)
        d_output = cuda.to_device(np.zeros((k_values.size, 4), dtype=np.float64))
        work_items = self._directions.shape[0] * k_values.size
        blocks = (work_items + self._threads_per_block - 1) // self._threads_per_block
        _quadratic_parts_cuda_kernel[blocks, self._threads_per_block](
            d_k_values,
            int(n),
            self._d_directions,
            self._d_weights,
            self._d_position,
            self._d_position_e,
            self._d_ell,
            self._source.orbit.f1,
            self._source.orbit.f2,
            d_output,
        )
        cuda.synchronize()
        return d_output.copy_to_host()


@dataclass(frozen=True)
class SourceHarmonic:
    """One harmonic and its analytic eccentricity derivative.

    The ``body_1`` and ``body_2`` fields include the corresponding mass
    fractions.  ``cross`` quantities are formed by callers from the two
    components, preserving the self/cross decomposition exactly.
    """

    total: complex
    total_e: complex
    body_1: complex
    body_1_e: complex
    body_2: complex
    body_2_e: complex


@dataclass(frozen=True)
class SourceSpectrum:
    """All nonnegative FFT coefficients at a fixed wave vector."""

    total: np.ndarray
    total_e: np.ndarray
    body_1: np.ndarray
    body_1_e: np.ndarray
    body_2: np.ndarray
    body_2_e: np.ndarray


class OffshellSource:
    """FFT source core using a uniform mean-anomaly grid.

    It implements the convention

    ``K_n(k) = <exp(+i*n*ell) S(ell,k)>``.

    NumPy's inverse FFT has precisely this positive-exponent convention when
    evaluated on the uniform mean-anomaly grid.
    """

    def __init__(self, orbit: BinaryOrbit, n_ell: int = 256) -> None:
        if n_ell < 16 or n_ell & (n_ell - 1):
            raise ValueError("n_ell must be a power of two and at least 16")
        self.orbit = orbit
        self.n_ell = int(n_ell)
        self.grid, self.position, self.position_e = relative_orbit_arrays(orbit, n_ell)

    def spectrum(self, k_vector: np.ndarray) -> SourceSpectrum:
        """Return all nonnegative harmonics at arbitrary ``k``."""

        k_vector = np.asarray(k_vector, dtype=np.float64)
        if k_vector.shape != (3,):
            raise ValueError("k_vector must have shape (3,)")
        phase = self.position @ k_vector
        phase_e = self.position_e @ k_vector
        body_1_samples = self.orbit.f1 * np.exp(-1j * self.orbit.f2 * phase)
        body_2_samples = self.orbit.f2 * np.exp(+1j * self.orbit.f1 * phase)
        body_1_e_samples = body_1_samples * (-1j * self.orbit.f2 * phase_e)
        body_2_e_samples = body_2_samples * (+1j * self.orbit.f1 * phase_e)

        body_1 = np.fft.ifft(body_1_samples)
        body_2 = np.fft.ifft(body_2_samples)
        body_1_e = np.fft.ifft(body_1_e_samples)
        body_2_e = np.fft.ifft(body_2_e_samples)
        return SourceSpectrum(
            total=body_1 + body_2,
            total_e=body_1_e + body_2_e,
            body_1=body_1,
            body_1_e=body_1_e,
            body_2=body_2,
            body_2_e=body_2_e,
        )

    def harmonic(self, n: int, k_vector: np.ndarray) -> SourceHarmonic:
        """Return ``K_n`` and ``K_n,e`` for one nonnegative harmonic."""

        if n < 0 or n > self.n_ell // 2:
            raise ValueError("n must satisfy 0 <= n <= n_ell/2")
        spectrum = self.spectrum(k_vector)
        return SourceHarmonic(
            total=complex(spectrum.total[n]),
            total_e=complex(spectrum.total_e[n]),
            body_1=complex(spectrum.body_1[n]),
            body_1_e=complex(spectrum.body_1_e[n]),
            body_2=complex(spectrum.body_2[n]),
            body_2_e=complex(spectrum.body_2_e[n]),
        )

    def direct_xi_harmonic(self, n: int, k_vector: np.ndarray, n_xi: int | None = None) -> SourceHarmonic:
        """Independent eccentric-anomaly quadrature used only for regression tests."""

        if n < 0:
            raise ValueError("n must be non-negative")
        n_xi = self.n_ell if n_xi is None else int(n_xi)
        if n_xi < 32:
            raise ValueError("n_xi must be at least 32")
        k_vector = np.asarray(k_vector, dtype=np.float64)
        if k_vector.shape != (3,):
            raise ValueError("k_vector must have shape (3,)")

        xi = 2.0 * math.pi * np.arange(n_xi, dtype=np.float64) / n_xi
        cos_xi = np.cos(xi)
        sin_xi = np.sin(xi)
        beta = math.sqrt(1.0 - self.orbit.e * self.orbit.e)
        jacobian = 1.0 - self.orbit.e * cos_xi
        ell = xi - self.orbit.e * sin_xi
        position = np.stack(
            (
                self.orbit.a * (cos_xi - self.orbit.e),
                self.orbit.a * beta * sin_xi,
                np.zeros_like(xi),
            ),
            axis=-1,
        )
        position_e = np.stack(
            (
                self.orbit.a * (-1.0 - sin_xi**2 / jacobian),
                self.orbit.a
                * (-self.orbit.e * sin_xi / beta + beta * cos_xi * sin_xi / jacobian),
                np.zeros_like(xi),
            ),
            axis=-1,
        )
        phase = position @ k_vector
        phase_e = position_e @ k_vector
        exp_harmonic = np.exp(1j * n * ell)
        body_1 = self.orbit.f1 * np.exp(-1j * self.orbit.f2 * phase)
        body_2 = self.orbit.f2 * np.exp(+1j * self.orbit.f1 * phase)
        body_1_e = body_1 * (-1j * self.orbit.f2 * phase_e)
        body_2_e = body_2 * (+1j * self.orbit.f1 * phase_e)

        def average(value: np.ndarray) -> complex:
            return complex(np.mean(jacobian * exp_harmonic * value))

        k1 = average(body_1)
        k2 = average(body_2)
        k1e = average(body_1_e)
        k2e = average(body_2_e)
        return SourceHarmonic(k1 + k2, k1e + k2e, k1, k1e, k2, k2e)

    def on_shell_harmonic(self, n: int, k: float, direction: np.ndarray) -> SourceHarmonic:
        """On-shell wrapper retained for flux-source regression tests."""

        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")
        norm = float(np.linalg.norm(direction))
        if not math.isclose(norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("direction must be a unit vector")
        return self.harmonic(n, float(k) * direction)

    def cuda_quadratic_source(self, directions: np.ndarray, weights: np.ndarray) -> CudaQuadraticSource:
        """Create a reusable CUDA angular evaluator for this source grid."""

        directions = np.asarray(directions, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape (n_angles, 3)")
        if weights.shape != (directions.shape[0],):
            raise ValueError("weights must have one entry per direction")
        return CudaQuadraticSource(self, directions, weights)


def source_quadratic_parts(value: SourceHarmonic) -> np.ndarray:
    """Return total, self-1, self-2, and cross ``Re[K^* K_e]`` parts."""

    total = float(np.real(np.conj(value.total) * value.total_e))
    self_1 = float(np.real(np.conj(value.body_1) * value.body_1_e))
    self_2 = float(np.real(np.conj(value.body_2) * value.body_2_e))
    return np.array((total, self_1, self_2, total - self_1 - self_2), dtype=np.float64)
