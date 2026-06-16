"""Quadrupole approximation for the classical-fluid n0=0 binary fluxes.

The normalized helpers match the conventions used by the numerical
calculators:

    P_norm = P / (2*rho_bar*M**2/c_s)
    tauOmega_norm = tau_z*tildeOmega / (2*rho_bar*M**2/c_s)

with the dimensionless parameter `A = a*tildeOmega/c_s`.  In these variables
the n0=0 quadrupole approximation is

    P_norm = (2*pi/15) * nu**2 * A**4
        * [7*sqrt(1-e**2) - 3*(1-e**2)] / (1-e**2)

    tauOmega_norm = (8*pi/15) * nu**2 * A**4 * sqrt(1-e**2).
"""

from __future__ import annotations

import argparse
import math

from classic_fluid_power import mass_fractions_from_nu


def _validate_nu_e_A(nu: float, e: float, A: float) -> None:
    mass_fractions_from_nu(nu)
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if A < 0.0:
        raise ValueError("A = a*tildeOmega/c_s must be non-negative")


def classical_quadrupole_power_eccentricity_factor(e: float) -> float:
    """Return `[7 sqrt(1-e^2) - 3(1-e^2)]/(1-e^2)`."""

    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    one_minus_e2 = 1.0 - e * e
    return (7.0 * math.sqrt(one_minus_e2) - 3.0 * one_minus_e2) / one_minus_e2


def classical_quadrupole_power_normalized(*, nu: float, e: float, A: float) -> float:
    """Return the normalized n0=0 quadrupole power.

    The returned value is `P/(2*rho_bar*M**2/c_s)`.
    """

    _validate_nu_e_A(nu, e, A)
    return (
        (2.0 * math.pi / 15.0)
        * nu
        * nu
        * A**4
        * classical_quadrupole_power_eccentricity_factor(e)
    )


def classical_quadrupole_tau_z_tildeOmega_normalized(
    *,
    nu: float,
    e: float,
    A: float,
) -> float:
    """Return the normalized n0=0 quadrupole z-torque.

    The returned value is `tau_z*tildeOmega/(2*rho_bar*M**2/c_s)`.
    """

    _validate_nu_e_A(nu, e, A)
    return (
        (8.0 * math.pi / 15.0)
        * nu
        * nu
        * A**4
        * math.sqrt(1.0 - e * e)
    )


def classical_quadrupole_power_physical(
    *,
    nu: float,
    e: float,
    Omega: float,
    a: float,
    M: float,
    rho_bar: float,
    c_s: float,
) -> float:
    """Return the physical quadrupole power from the user's formula."""

    mass_fractions_from_nu(nu)
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if Omega < 0.0 or a < 0.0 or M < 0.0 or rho_bar < 0.0 or c_s <= 0.0:
        raise ValueError("Omega, a, M, rho_bar must be non-negative and c_s positive")
    return (
        (4.0 * math.pi / 15.0)
        * nu
        * nu
        * M
        * M
        * rho_bar
        * Omega**4
        * a**4
        / c_s
        * classical_quadrupole_power_eccentricity_factor(e)
    )


def classical_quadrupole_tau_z_physical(
    *,
    nu: float,
    e: float,
    Omega: float,
    a: float,
    M: float,
    rho_bar: float,
    c_s: float,
) -> float:
    """Return the physical quadrupole z-torque from the user's formula."""

    mass_fractions_from_nu(nu)
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if Omega < 0.0 or a < 0.0 or M < 0.0 or rho_bar < 0.0 or c_s <= 0.0:
        raise ValueError("Omega, a, M, rho_bar must be non-negative and c_s positive")
    return (
        (16.0 * math.pi / 15.0)
        * nu
        * nu
        * M
        * M
        * rho_bar
        * Omega**3
        * a**4
        / (c_s * c_s)
        * math.sqrt(1.0 - e * e)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the n0=0 classical-fluid quadrupole approximation."
    )
    parser.add_argument("--nu", type=float, required=True, help="nu = m1*m2/M^2")
    parser.add_argument("--e", type=float, required=True, help="orbital eccentricity")
    parser.add_argument("--A", type=float, required=True, help="A = a*tildeOmega/c_s")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_norm = classical_quadrupole_power_normalized(nu=args.nu, e=args.e, A=args.A)
    tau_norm = classical_quadrupole_tau_z_tildeOmega_normalized(
        nu=args.nu,
        e=args.e,
        A=args.A,
    )
    print(f"quadrupole_power_normalized = {p_norm:.16e}")
    print(f"quadrupole_tau_z_tildeOmega_normalized = {tau_norm:.16e}")


if __name__ == "__main__":
    main()
