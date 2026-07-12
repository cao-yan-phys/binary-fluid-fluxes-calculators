"""General quadrupole-approximation flux calculators.

This module implements the Bessel-expression quadrupole sums without setting
``n0 = 0``.

The normalized helpers match the existing full numerical calculators:

Classical fluid:
    ``P/(2*rho_bar*M**2/c_s)`` and
    ``tau_z*tildeOmega/(2*rho_bar*M**2/c_s)``,
    with ``A = a*tildeOmega/c_s``.

Quantum/SP fluid with ``c_S=0``:
    ``P/(2*rho_bar*M**2*m_phi/sqrt(Omega))`` and
    ``tau_z*tildeOmega/(2*rho_bar*M**2*m_phi/sqrt(Omega))``,
    with ``Omega = 2*m_phi*tildeOmega`` and ``A = a*sqrt(Omega)``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import jv, jvp

from classic_fluid_power import mass_fractions_from_nu


Medium = Literal["classical", "quantum"]
InvariantMethod = Literal["bessel", "direct", "auto"]


@dataclass(frozen=True)
class QuadrupoleFluxResult:
    medium: str
    P: float
    tau_z_tildeOmega: float
    n_values: np.ndarray
    P_terms: np.ndarray
    tau_terms: np.ndarray
    converged: bool
    tail_ratio_P: float
    tail_ratio_tau_z: float
    n0: float
    A: float
    nu: float
    e: float
    n_max_evaluated: int
    invariant_method: str
    max_ak_99pct_power: float
    quadrupole_warning: str | None

    def to_json_dict(self) -> dict[str, float | int | bool | str | None]:
        out = asdict(self)
        out.pop("n_values")
        out.pop("P_terms")
        out.pop("tau_terms")
        return out


def _validate_nu_e_n0_A(*, nu: float, e: float, n0: float, A: float) -> None:
    mass_fractions_from_nu(nu)
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if n0 < 0.0:
        raise ValueError("n0 must be non-negative")
    if A < 0.0:
        raise ValueError("A must be non-negative")


def quadrupole_invariants_direct(
    n: int,
    e: float,
    *,
    a: float = 1.0,
    n_xi: int = 32768,
) -> tuple[float, float]:
    """Return ``(S_n, L_n)`` by direct xi quadrature."""

    if n < 1:
        raise ValueError("n must be at least 1")
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if a < 0.0:
        raise ValueError("a must be non-negative")
    if n_xi < 64:
        raise ValueError("n_xi must be at least 64")

    xi = 2.0 * math.pi * (np.arange(n_xi, dtype=np.float64) + 0.5) / n_xi
    cos_xi = np.cos(xi)
    sin_xi = np.sin(xi)
    jac = 1.0 - e * cos_xi
    phase = n * (xi - e * sin_xi)
    exp_phase = np.exp(1j * phase)
    x = a * (cos_xi - e)
    y = a * math.sqrt(1.0 - e * e) * sin_xi

    weight = jac * exp_phase
    i_xx = np.mean(weight * x * x)
    i_yy = np.mean(weight * y * y)
    i_xy = np.mean(weight * x * y)

    trace = i_xx + i_yy
    iij_iij = abs(i_xx) ** 2 + abs(i_yy) ** 2 + 2.0 * abs(i_xy) ** 2
    s_n = abs(trace) ** 2 + 2.0 * iij_iij
    l_complex = -1j * (
        np.conj(i_xx) * i_xy
        + np.conj(i_xy) * i_yy
        - np.conj(i_xy) * i_xx
        - np.conj(i_yy) * i_xy
    )
    return float(s_n.real), float(l_complex.real)


def quadrupole_invariants(
    n: int,
    e: float,
    *,
    a: float = 1.0,
    method: InvariantMethod = "bessel",
    e_small: float = 1.0e-6,
    n_xi: int = 32768,
) -> tuple[float, float]:
    """Return the real quadrupole invariants ``(S_n, L_n)``.

    ``S_n`` and ``L_n`` include the factor ``a**4``.  For ``e < e_small`` the
    circular fallback is used to avoid cancellation in the Bessel formula.
    """

    if n < 1:
        raise ValueError("n must be at least 1")
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if a < 0.0:
        raise ValueError("a must be non-negative")
    if method not in ("bessel", "direct", "auto"):
        raise ValueError("method must be 'bessel', 'direct', or 'auto'")

    if e < e_small:
        if n == 2:
            return 0.5 * a**4, 0.25 * a**4
        return 0.0, 0.0
    if method == "direct":
        return quadrupole_invariants_direct(n, e, a=a, n_xi=n_xi)

    e2 = e * e
    n_float = float(n)
    z = n_float * e
    j = float(jv(n, z))
    jp = float(jvp(n, z, 1))
    prefactor = a**4 / (e**4 * n_float**4)

    s_bracket = (
        4.0
        * e2
        * (e2 - 1.0)
        * (((e2 - 1.0) * n_float * n_float) - 1.0)
        * jp
        * jp
        - 4.0
        * e
        * (3.0 * e2 * e2 - 7.0 * e2 + 4.0)
        * n_float
        * j
        * jp
        + (
            3.0 * e2 * e2
            - 4.0 * (e2 - 1.0) ** 3 * n_float * n_float
            - 4.0 * e2
            + 4.0
        )
        * j
        * j
    )
    s_n = 4.0 * prefactor * s_bracket

    l_n = (
        8.0
        * prefactor
        * math.sqrt(1.0 - e2)
        * (((e2 - 1.0) * n_float * j) + e * jp)
        * (2.0 * e * (e2 - 1.0) * n_float * jp - (e2 - 2.0) * j)
    )
    return float(s_n), float(l_n)


def _terms_for_medium(
    *,
    medium: Medium,
    nu: float,
    e: float,
    n0: float,
    A: float,
    n_values: np.ndarray,
    invariant_method: InvariantMethod,
    e_small: float,
    direct_n_xi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_values = np.empty(n_values.size, dtype=np.float64)
    l_values = np.empty(n_values.size, dtype=np.float64)
    for i, n in enumerate(n_values):
        s_values[i], l_values[i] = quadrupole_invariants(
            int(n),
            e,
            a=1.0,
            method=invariant_method,
            e_small=e_small,
            n_xi=direct_n_xi,
        )

    n_float = n_values.astype(np.float64)
    n2_plus_n02 = n_float * n_float + n0 * n0
    if medium == "classical":
        weight = n2_plus_n02 ** 1.5
        ak_values = A * np.sqrt(n2_plus_n02)
        p_terms = (math.pi / 15.0) * nu * nu * A**4 * n_float * weight * s_values
        tau_terms = (4.0 * math.pi / 15.0) * nu * nu * A**4 * weight * l_values
    elif medium == "quantum":
        weight = n2_plus_n02 ** 0.25
        ak_values = A * np.sqrt(np.sqrt(n2_plus_n02))
        p_terms = (math.pi / 15.0) * nu * nu * A**4 * n_float * weight * s_values
        tau_terms = (4.0 * math.pi / 15.0) * nu * nu * A**4 * weight * l_values
    else:
        raise ValueError("medium must be 'classical' or 'quantum'")
    return p_terms, tau_terms, ak_values


def _max_ak_for_dominant_power(
    p_terms: np.ndarray,
    ak_values: np.ndarray,
    *,
    fraction: float = 0.99,
) -> float:
    total = float(np.sum(np.abs(p_terms)))
    if total <= 0.0:
        return 0.0
    cumulative = np.cumsum(np.abs(p_terms))
    stop = int(np.searchsorted(cumulative, fraction * total, side="left"))
    stop = min(stop, ak_values.size - 1)
    return float(np.max(ak_values[: stop + 1]))


def _quadrupole_warning(max_ak_99: float, warning_ak: float) -> str | None:
    if max_ak_99 <= warning_ak:
        return None
    return (
        f"quadrupole expansion may be outside its controlled regime: "
        f"max(a*k_n) over 99% of power is {max_ak_99:.6g} > {warning_ak:.6g}"
    )


def quadrupole_flux_normalized(
    *,
    medium: Medium,
    nu: float,
    e: float,
    n0: float,
    A: float,
    n_max: int = 4096,
    rtol: float = 1.0e-10,
    tail_window: int = 32,
    consecutive_windows: int = 3,
    chunk_size: int = 64,
    invariant_method: InvariantMethod = "bessel",
    e_small: float = 1.0e-6,
    direct_n_xi: int = 32768,
    strict_convergence: bool = True,
    warning_ak: float = 0.3,
) -> QuadrupoleFluxResult:
    """Compute normalized ``P`` and ``tau_z*tildeOmega`` in the quadrupole approximation."""

    _validate_nu_e_n0_A(nu=nu, e=e, n0=n0, A=A)
    if medium not in ("classical", "quantum"):
        raise ValueError("medium must be 'classical' or 'quantum'")
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    if tail_window < 1:
        raise ValueError("tail_window must be at least 1")
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    all_n: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_tau: list[np.ndarray] = []
    all_ak: list[np.ndarray] = []
    converged = False
    convergence_passes = 0
    tail_ratio_p = math.inf
    tail_ratio_tau = math.inf

    for start in range(1, n_max + 1, chunk_size):
        stop = min(n_max, start + chunk_size - 1)
        n_values = np.arange(start, stop + 1, dtype=np.int32)
        p_terms, tau_terms, ak_values = _terms_for_medium(
            medium=medium,
            nu=nu,
            e=e,
            n0=n0,
            A=A,
            n_values=n_values,
            invariant_method=invariant_method,
            e_small=e_small,
            direct_n_xi=direct_n_xi,
        )
        all_n.append(n_values)
        all_p.append(p_terms)
        all_tau.append(tau_terms)
        all_ak.append(ak_values)

        p_flat = np.concatenate(all_p)
        tau_flat = np.concatenate(all_tau)
        if p_flat.size >= tail_window:
            p_total = float(np.sum(p_flat))
            tau_total = float(np.sum(tau_flat))
            p_tail = float(np.sum(np.abs(p_flat[-tail_window:])))
            tau_tail = float(np.sum(np.abs(tau_flat[-tail_window:])))
            tail_ratio_p = p_tail / max(abs(p_total), np.finfo(np.float64).tiny)
            tail_ratio_tau = tau_tail / max(abs(tau_total), np.finfo(np.float64).tiny)
            if max(tail_ratio_p, tail_ratio_tau) <= rtol:
                convergence_passes += 1
                if convergence_passes >= consecutive_windows:
                    converged = True
                    break
            else:
                convergence_passes = 0

    n_done = np.concatenate(all_n)
    p_done = np.concatenate(all_p)
    tau_done = np.concatenate(all_tau)
    ak_done = np.concatenate(all_ak)
    if not math.isfinite(tail_ratio_p):
        tail_count = min(tail_window, p_done.size)
        tail_ratio_p = float(np.sum(np.abs(p_done[-tail_count:]))) / max(
            abs(float(np.sum(p_done))),
            np.finfo(np.float64).tiny,
        )
        tail_ratio_tau = float(np.sum(np.abs(tau_done[-tail_count:]))) / max(
            abs(float(np.sum(tau_done))),
            np.finfo(np.float64).tiny,
        )

    if strict_convergence and not converged:
        raise RuntimeError(
            f"{medium} quadrupole harmonic sum did not converge before n_max={n_max}; "
            f"tail_ratio_P={tail_ratio_p:.6e}, tail_ratio_tau={tail_ratio_tau:.6e}"
        )

    max_ak_99 = _max_ak_for_dominant_power(p_done, ak_done)
    return QuadrupoleFluxResult(
        medium=medium,
        P=float(np.sum(p_done)),
        tau_z_tildeOmega=float(np.sum(tau_done)),
        n_values=n_done,
        P_terms=p_done,
        tau_terms=tau_done,
        converged=bool(converged),
        tail_ratio_P=float(tail_ratio_p),
        tail_ratio_tau_z=float(tail_ratio_tau),
        n0=float(n0),
        A=float(A),
        nu=float(nu),
        e=float(e),
        n_max_evaluated=int(n_done[-1]),
        invariant_method=str(invariant_method),
        max_ak_99pct_power=max_ak_99,
        quadrupole_warning=_quadrupole_warning(max_ak_99, warning_ak),
    )


def classical_quadrupole_flux_normalized(**kwargs) -> QuadrupoleFluxResult:
    return quadrupole_flux_normalized(medium="classical", **kwargs)


def quantum_quadrupole_flux_normalized(**kwargs) -> QuadrupoleFluxResult:
    return quadrupole_flux_normalized(medium="quantum", **kwargs)


def classical_quadrupole_flux_physical(
    *,
    a: float,
    e: float,
    nu: float,
    M: float,
    rho_bar: float,
    c_s: float,
    Omega_phys: float,
    n0: float | None = None,
    **kwargs,
) -> tuple[float, float, QuadrupoleFluxResult]:
    """Return physical ``(P, tau_z, normalized_result)`` for the classical fluid."""

    if c_s <= 0.0 or Omega_phys <= 0.0:
        raise ValueError("c_s and Omega_phys must be positive")
    if a < 0.0 or M < 0.0 or rho_bar < 0.0:
        raise ValueError("a, M, and rho_bar must be non-negative")
    omega_aux = Omega_phys / c_s
    if n0 is None:
        m_j = math.sqrt(4.0 * math.pi * rho_bar) / c_s
        n0 = m_j / omega_aux
    result = classical_quadrupole_flux_normalized(
        nu=nu,
        e=e,
        n0=n0,
        A=a * omega_aux,
        **kwargs,
    )
    norm = 2.0 * rho_bar * M * M / c_s
    return result.P * norm, result.tau_z_tildeOmega * norm / Omega_phys, result


def quantum_quadrupole_flux_physical(
    *,
    a: float,
    e: float,
    nu: float,
    M: float,
    rho_bar: float,
    m_phi: float,
    Omega_phys: float,
    n0: float | None = None,
    **kwargs,
) -> tuple[float, float, QuadrupoleFluxResult]:
    """Return physical ``(P, tau_z, normalized_result)`` for the quantum/SP fluid."""

    if m_phi <= 0.0 or Omega_phys <= 0.0:
        raise ValueError("m_phi and Omega_phys must be positive")
    if a < 0.0 or M < 0.0 or rho_bar < 0.0:
        raise ValueError("a, M, and rho_bar must be non-negative")
    omega_aux = 2.0 * m_phi * Omega_phys
    if n0 is None:
        m_q = math.sqrt(16.0 * math.pi * m_phi * m_phi * rho_bar)
        n0 = m_q / omega_aux
    result = quantum_quadrupole_flux_normalized(
        nu=nu,
        e=e,
        n0=n0,
        A=a * math.sqrt(omega_aux),
        **kwargs,
    )
    norm = 2.0 * rho_bar * M * M * m_phi / math.sqrt(omega_aux)
    return result.P * norm, result.tau_z_tildeOmega * norm / Omega_phys, result


def _write_terms_csv(path: Path, result: QuadrupoleFluxResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        (
            result.n_values,
            result.P_terms,
            np.cumsum(result.P_terms),
            result.tau_terms,
            np.cumsum(result.tau_terms),
        )
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="n,P_term,P_cumulative,tau_z_tildeOmega_term,tau_z_tildeOmega_cumulative",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium", choices=("classical", "quantum"), required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--e", type=float, required=True)
    parser.add_argument("--n0", type=float, required=True)
    parser.add_argument("--A", type=float, required=True)
    parser.add_argument("--n-max", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--tail-window", type=int, default=32)
    parser.add_argument("--consecutive-windows", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--method", choices=("bessel", "direct", "auto"), default="bessel")
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--save-terms", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = quadrupole_flux_normalized(
        medium=args.medium,
        nu=args.nu,
        e=args.e,
        n0=args.n0,
        A=args.A,
        n_max=args.n_max,
        rtol=args.rtol,
        tail_window=args.tail_window,
        consecutive_windows=args.consecutive_windows,
        chunk_size=args.chunk_size,
        invariant_method=args.method,
        strict_convergence=not args.allow_unconverged,
    )
    print(f"medium = {result.medium}")
    print(f"P_normalized = {result.P:.16e}")
    print(f"tau_z_tildeOmega_normalized = {result.tau_z_tildeOmega:.16e}")
    print(f"P_over_tau_z_tildeOmega = {result.P / result.tau_z_tildeOmega:.16e}")
    print(f"converged = {result.converged}")
    print(f"n_evaluated = 1..{result.n_max_evaluated}")
    print(f"tail_ratio_P = {result.tail_ratio_P:.16e}")
    print(f"tail_ratio_tau_z = {result.tail_ratio_tau_z:.16e}")
    print(f"max_ak_99pct_power = {result.max_ak_99pct_power:.16e}")
    if result.quadrupole_warning:
        print(f"warning = {result.quadrupole_warning}")
    if args.save_terms is not None:
        _write_terms_csv(args.save_terms, result)
        print(f"terms_csv = {args.save_terms}")
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(result.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"json = {args.save_json}")


if __name__ == "__main__":
    main()
