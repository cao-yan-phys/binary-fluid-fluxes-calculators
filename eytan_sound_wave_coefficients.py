"""Finite-cutoff Eytan--Desjacques--Ginat single-perturber coefficient calculator.

This module implements the eccentric single-perturber harmonic coefficients

    I_E = A * sum_{j != 0, l, m} c_lm |g_lm^(j)(j*A, e)|^2,
    I_L = A * sum_{j != 0, l, m} (m/j) c_lm |g_lm^(j)(j*A, e)|^2,

with finite cutoffs ``jmax`` and ``lmax``.

Here ``A = a*Omega/c_s`` for the fixed-center single perturber.  The returned
``P_shape`` and ``tau_z_shape`` are not the normalized fluxes themselves.  For
perturber mass ``m_p``,

    P / (2*rho_bar*m_p**2/c_s) = 2*pi*P_shape,
    tau_z*tildeOmega / (2*rho_bar*m_p**2/c_s) = 2*pi*A*tau_z_shape.

The returned shape fields are defined as

    P_shape = edot_shape = I_E / A,
    tau_z_shape = ldot_shape = I_L / A^2,

for ``a = c_s = rho = m_p = 1``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, lpmv, spherical_jn


@dataclass(frozen=True)
class EytanSoundWaveResult:
    IE: float
    IL: float
    P_shape: float
    tau_z_shape: float
    edot_shape: float
    ldot_shape: float
    A: float
    e: float
    jmax: int
    lmax: int
    n_xi: int


def _validate_inputs(*, A: float, e: float, jmax: int, lmax: int, n_xi: int) -> None:
    if A <= 0.0:
        raise ValueError("A must be positive")
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if jmax < 1:
        raise ValueError("jmax must be at least 1")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if n_xi < 64:
        raise ValueError("n_xi must be at least 64")


def c_lm(l: int, m: int) -> float:
    """Return the coefficient c_lm, invariant under m -> -m."""

    ma = abs(int(m))
    if ma > l:
        return 0.0
    p_lm_0 = float(lpmv(ma, l, 0.0))
    if p_lm_0 == 0.0:
        return 0.0
    log_factorial_ratio = gammaln(l - ma + 1.0) - gammaln(l + ma + 1.0)
    return float((2 * l + 1) * math.exp(log_factorial_ratio) * p_lm_0 * p_lm_0)


def _orbit_arrays(e: float, n_xi: int) -> tuple[np.ndarray, ...]:
    xi = 2.0 * math.pi * (np.arange(n_xi, dtype=np.float64) + 0.5) / n_xi
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)
    mean_anomaly = xi - e * sin_xi
    jacobian = 1.0 - e * cos_xi
    radius_over_a = jacobian
    true_anomaly = np.arctan2(math.sqrt(1.0 - e * e) * sin_xi, cos_xi - e)
    return mean_anomaly, jacobian, radius_over_a, true_anomaly


def eytan_sound_wave_coefficients(
    *,
    A: float,
    e: float,
    jmax: int = 20,
    lmax: int = 13,
    n_xi: int = 8192,
) -> EytanSoundWaveResult:
    """Compute finite-cutoff ``I_E``, ``I_L``, and the corresponding rate shapes."""

    _validate_inputs(A=A, e=e, jmax=jmax, lmax=lmax, n_xi=n_xi)
    mean_anomaly, jacobian, radius_over_a, true_anomaly = _orbit_arrays(e, n_xi)

    ie_sum = 0.0
    il_sum = 0.0
    for j in range(-jmax, jmax + 1):
        if j == 0:
            continue
        x = j * A
        x_abs = abs(x)
        exp_minus_ijM = np.exp(-1j * j * mean_anomaly)
        for ell in range(0, lmax + 1):
            sign = -1.0 if (x < 0.0 and (ell % 2 == 1)) else 1.0
            jl = sign * spherical_jn(ell, x_abs * radius_over_a)
            base = jacobian * jl * exp_minus_ijM
            for m in range(-ell, ell + 1):
                coeff = c_lm(ell, m)
                if coeff == 0.0:
                    continue
                phase = np.exp(1j * m * true_anomaly)
                g = np.mean(base * phase)
                g2 = float((g.real * g.real) + (g.imag * g.imag))
                weighted = coeff * g2
                ie_sum += weighted
                il_sum += (m / j) * weighted

    IE = float(A * ie_sum)
    IL = float(A * il_sum)
    P_shape = IE / A
    tau_z_shape = IL / (A * A)
    return EytanSoundWaveResult(
        IE=IE,
        IL=IL,
        P_shape=P_shape,
        tau_z_shape=tau_z_shape,
        edot_shape=P_shape,
        ldot_shape=tau_z_shape,
        A=A,
        e=e,
        jmax=int(jmax),
        lmax=int(lmax),
        n_xi=int(n_xi),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the finite-cutoff Eytan--Desjacques--Ginat "
            "single-perturber coefficient calculator."
        )
    )
    parser.add_argument("--A", type=float, required=True)
    parser.add_argument("--e", type=float, required=True)
    parser.add_argument("--jmax", type=int, default=20)
    parser.add_argument("--lmax", type=int, default=13)
    parser.add_argument("--n-xi", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = eytan_sound_wave_coefficients(
        A=args.A,
        e=args.e,
        jmax=args.jmax,
        lmax=args.lmax,
        n_xi=args.n_xi,
    )
    print(f"IE = {result.IE:.16e}")
    print(f"IL = {result.IL:.16e}")
    print(f"edot_shape_IE_over_A = {result.edot_shape:.16e}")
    print(f"ldot_shape_IL_over_A2 = {result.ldot_shape:.16e}")


if __name__ == "__main__":
    main()
