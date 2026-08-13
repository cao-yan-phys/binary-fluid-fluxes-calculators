"""Cached pair geometry on a uniform mean-anomaly grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .orbit_derivatives import BinaryOrbit, KeplerGrid, relative_orbit_arrays


@dataclass(frozen=True)
class PairGeometry:
    """Separation and fixed-mean-anomaly eccentricity derivative for one pair."""

    separation: np.ndarray
    separation_e: np.ndarray


@dataclass(frozen=True)
class PairOrbitGeometry:
    """All four center-of-mass pair geometries for a binary orbit."""

    grid: KeplerGrid
    delta: np.ndarray
    pairs: dict[tuple[int, int], PairGeometry]
    mass_fractions: tuple[float, float]

    def derivative_parts(
        self,
        radial_derivative: callable,
        phase: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Return total, body-1, body-2, and cross pair contributions.

        ``radial_derivative`` is evaluated separately on every pair separation,
        so the cross term is accumulated directly rather than reconstructed by
        subtraction.
        """

        fractions = self.mass_fractions
        parts = np.zeros(4, dtype=np.float64)
        for body_a in range(2):
            for body_b in range(2):
                pair = self.pairs[(body_a, body_b)]
                contribution = fractions[body_a] * fractions[body_b] * float(
                    np.mean(np.asarray(phase) * radial_derivative(pair.separation) * pair.separation_e)
                )
                parts[0] += contribution
                if body_a == 0 and body_b == 0:
                    parts[1] += contribution
                elif body_a == 1 and body_b == 1:
                    parts[2] += contribution
                else:
                    parts[3] += contribution
        return parts


def build_pair_orbit_geometry(orbit: BinaryOrbit, n_ell: int) -> PairOrbitGeometry:
    """Build reusable pair matrices and analytic ``partial_e r`` values."""

    grid, position, position_e = relative_orbit_arrays(orbit, n_ell)
    alpha = (orbit.f2, -orbit.f1)
    bodies = tuple(alpha_value * position for alpha_value in alpha)
    bodies_e = tuple(alpha_value * position_e for alpha_value in alpha)
    pairs: dict[tuple[int, int], PairGeometry] = {}
    for body_a in range(2):
        for body_b in range(2):
            vector = bodies[body_a][:, None, :] - bodies[body_b][None, :, :]
            vector_e = bodies_e[body_a][:, None, :] - bodies_e[body_b][None, :, :]
            separation = np.linalg.norm(vector, axis=-1)
            numerator = np.sum(vector * vector_e, axis=-1)
            separation_e = np.divide(
                numerator,
                separation,
                out=np.zeros_like(separation),
                where=separation > 0.0,
            )
            pairs[(body_a, body_b)] = PairGeometry(separation=separation, separation_e=separation_e)
    delta = grid.ell[:, None] - grid.ell[None, :]
    return PairOrbitGeometry(
        grid=grid,
        delta=delta,
        pairs=pairs,
        mass_fractions=(orbit.f1, orbit.f2),
    )
