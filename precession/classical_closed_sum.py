
from __future__ import annotations

import math

import numpy as np


TWO_PI = 2.0 * math.pi


def s2_periodic(argument: np.ndarray | float) -> np.ndarray:

    reduced = np.remainder(np.asarray(argument, dtype=np.float64), TWO_PI)
    return math.pi * math.pi / 6.0 - 0.5 * math.pi * reduced + 0.25 * reduced * reduced


def _image_sum(delta: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    lower = int(math.floor(float(np.min(delta - y)) / TWO_PI))
    upper = int(math.ceil(float(np.max(delta + y)) / TWO_PI))
    triangles = np.zeros_like(y)
    distances = np.zeros_like(y)
    for image in range(lower, upper + 1):
        distance = np.abs(delta - TWO_PI * image)
        active = distance < y
        triangles += np.where(active, y - distance, 0.0)
        distances += np.where(active, distance, 0.0)
    return triangles, distances


def classical_total_pair_kernel(
    delta: np.ndarray | float,
    radius: np.ndarray | float,
    *,
    rho_bar: float,
    omega_tilde: float,
    c_s: float,
) -> np.ndarray:

    r, difference = np.broadcast_arrays(np.asarray(radius, dtype=np.float64), np.asarray(delta, dtype=np.float64))
    y = omega_tilde * r / c_s
    triangles, _ = _image_sum(difference, y)
    prefactor = 4.0 * math.pi * math.pi * rho_bar / (omega_tilde * omega_tilde)
    kernel = np.divide(prefactor * triangles, r, out=np.zeros_like(r), where=r > 0.0)
    zero = r == 0.0
    if np.any(zero):
        kernel = np.where(zero, prefactor * omega_tilde / c_s, kernel)
    return kernel


def classical_total_pair_kernel_dr(
    delta: np.ndarray | float,
    radius: np.ndarray | float,
    *,
    rho_bar: float,
    omega_tilde: float,
    c_s: float,
) -> np.ndarray:

    r, difference = np.broadcast_arrays(np.asarray(radius, dtype=np.float64), np.asarray(delta, dtype=np.float64))
    y = omega_tilde * r / c_s
    _, distances = _image_sum(difference, y)
    prefactor = 4.0 * math.pi * math.pi * rho_bar / (omega_tilde * omega_tilde)
    return np.divide(prefactor * distances, r * r, out=np.zeros_like(r), where=r > 0.0)
