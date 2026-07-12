"""Calculators for the quantum-fluid normalized observables.

The default quantum-fluid formulas use

    a k_n = A * (n**2 + n0**2)**(1/4),    A = a * sqrt(Omega),

where `n0 = m/Omega`.  This differs from the classical-fluid convention
`A = a*Omega`.

For a finite quartic-interaction sound term, pass the signed parameter
`cS2_over_Omega = c_S**2/Omega`.  The convenience parameter
`cS_over_sqrtOmega = c_S/sqrt(Omega)` is also accepted for `c_S**2 >= 0`.
Then

    a k_n = A * kappa_n,
    kappa_n**2 = (-S + sqrt(S**2 + 4*(n**2+n0**2))) / 2,
    S = cS2_over_Omega.

Returned normalizations:

    power:
        P / (2 * rho_bar * M**2 * m_phi / sqrt(Omega))

    tau_z:
        tau_z * tilde_Omega
        / (2 * rho_bar * M**2 * m_phi / sqrt(Omega))

    force_y:
        F_y * sqrt(Omega) / m_phi
        / (2 * rho_bar * M**2 * m_phi / sqrt(Omega))

The force is along y for the setup in Eq. (2.52), so `force_y` is the nonzero
component of `F_q`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numba import cuda, njit, prange

from classic_fluid_power import (
    DEFAULT_CONSECUTIVE_WINDOWS,
    DEFAULT_MAX_N,
    DEFAULT_RTOL,
    DEFAULT_TAIL_WINDOW,
    TWO_PI,
    ClassicalFluidResult,
    ConvergenceError,
    build_quadrature,
    mass_fractions_from_nu,
    recommended_n_xi,
)


Backend = Literal["auto", "cuda", "cpu"]
Quantity = Literal["power", "tau_z", "force_y"]

QUANTITY_INDEX = {
    "power": 0,
    "tau_z": 1,
    "force_y": 2,
}


def _cuda_available() -> bool:
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _validate_inputs(
    *,
    nu: float,
    e: float,
    n0: float,
    A: float,
    n_max: int,
    n_xi: int | None,
    n_mu: int,
    n_phi: int,
    cS2_over_Omega: float,
) -> None:
    mass_fractions_from_nu(nu)
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if n0 < 0.0:
        raise ValueError("n0 must be non-negative")
    if A < 0.0:
        raise ValueError("A = a*sqrt(Omega) must be non-negative")
    if not math.isfinite(cS2_over_Omega):
        raise ValueError("cS2_over_Omega = c_S^2/Omega must be finite")
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    if n_xi is not None and n_xi < 8:
        raise ValueError("n_xi must be at least 8")
    if n_mu < 2:
        raise ValueError("n_mu must be at least 2")
    if n_phi < 4:
        raise ValueError("n_phi must be at least 4")


@njit(parallel=True, fastmath=True)
def _quantum_terms_cpu(
    n_values: np.ndarray,
    mu: np.ndarray,
    w_mu: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    cos_xi: np.ndarray,
    sin_xi: np.ndarray,
    xi_minus_e_sin_xi: np.ndarray,
    q1: float,
    q2: float,
    e: float,
    sqrt_one_minus_e2: float,
    A: float,
    n0: float,
    cS2_over_Omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = np.zeros(n_values.size, dtype=np.float64)
    tau_z = np.zeros(n_values.size, dtype=np.float64)
    force_y = np.zeros(n_values.size, dtype=np.float64)
    n_phi = cos_phi.size
    n_xi = cos_xi.size
    phi_weight = TWO_PI / n_phi

    for i_n in prange(n_values.size):
        n_float = float(n_values[i_n])
        n2_plus_n02 = n_float * n_float + n0 * n0
        d = math.sqrt(cS2_over_Omega * cS2_over_Omega + 4.0 * n2_plus_n02)
        kappa2 = 0.5 * (d - cS2_over_Omega)
        kappa = math.sqrt(kappa2)
        ak = A * kappa
        common_weight = 2.0 / (kappa * d)
        power_weight = n_float * common_weight
        tau_weight = common_weight
        force_weight = 4.0 / d

        power_sum = 0.0
        tau_sum = 0.0
        force_sum = 0.0

        for i_mu in range(mu.size):
            mu_i = mu[i_mu]
            sin_theta_sq = 1.0 - mu_i * mu_i
            if sin_theta_sq < 0.0:
                sin_theta_sq = 0.0
            sin_theta = math.sqrt(sin_theta_sq)
            mu_weight = w_mu[i_mu]

            for i_phi in range(n_phi):
                cp = cos_phi[i_phi]
                sp = sin_phi[i_phi]
                y_component = sin_theta * sp
                k_re = 0.0
                k_im = 0.0
                dk_re = 0.0
                dk_im = 0.0

                for i_xi in range(n_xi):
                    cxi = cos_xi[i_xi]
                    sxi = sin_xi[i_xi]
                    dot = sin_theta * (
                        cp * (cxi - e) + sp * sqrt_one_minus_e2 * sxi
                    )
                    dot_phi = sin_theta * (
                        -sp * (cxi - e) + cp * sqrt_one_minus_e2 * sxi
                    )
                    z = ak * dot
                    z_phi = ak * dot_phi
                    time_phase = n_float * xi_minus_e_sin_xi[i_xi]

                    base_re = math.cos(time_phase)
                    base_im = math.sin(time_phase)
                    sin_q1 = math.sin(q1 * z)
                    cos_q1 = math.cos(q1 * z)
                    sin_q2 = math.sin(q2 * z)
                    cos_q2 = math.cos(q2 * z)

                    bracket_re = q1 * cos_q2 + q2 * cos_q1
                    bracket_im = -q1 * sin_q2 + q2 * sin_q1
                    dbracket_re = -q1 * q2 * z_phi * (sin_q1 + sin_q2)
                    dbracket_im = q1 * q2 * z_phi * (cos_q1 - cos_q2)
                    jac = 1.0 - e * cxi

                    k_re += jac * (base_re * bracket_re - base_im * bracket_im)
                    k_im += jac * (base_im * bracket_re + base_re * bracket_im)
                    dk_re += jac * (base_re * dbracket_re - base_im * dbracket_im)
                    dk_im += jac * (base_im * dbracket_re + base_re * dbracket_im)

                k_re /= n_xi
                k_im /= n_xi
                dk_re /= n_xi
                dk_im /= n_xi
                k_abs2 = k_re * k_re + k_im * k_im
                torque_density = k_re * dk_im - k_im * dk_re
                angle_weight = mu_weight * phi_weight

                power_sum += angle_weight * k_abs2
                tau_sum += angle_weight * torque_density
                force_sum += angle_weight * y_component * k_abs2

        power[i_n] = power_weight * power_sum
        tau_z[i_n] = tau_weight * tau_sum
        force_y[i_n] = force_weight * force_sum

    return power, tau_z, force_y


@cuda.jit
def _quantum_terms_cuda_kernel(
    n_values,
    mu,
    w_mu,
    cos_phi,
    sin_phi,
    cos_xi,
    sin_xi,
    xi_minus_e_sin_xi,
    q1,
    q2,
    e,
    sqrt_one_minus_e2,
    A,
    n0,
    cS2_over_Omega,
    out_power,
    out_tau_z,
    out_force_y,
):
    idx = cuda.grid(1)
    n_mu = mu.size
    n_phi = cos_phi.size
    n_xi = cos_xi.size
    n_angles = n_mu * n_phi
    total = n_values.size * n_angles
    if idx >= total:
        return

    i_n = idx // n_angles
    rem = idx - i_n * n_angles
    i_mu = rem // n_phi
    i_phi = rem - i_mu * n_phi

    n_float = float(n_values[i_n])
    n2_plus_n02 = n_float * n_float + n0 * n0
    d = math.sqrt(cS2_over_Omega * cS2_over_Omega + 4.0 * n2_plus_n02)
    kappa2 = 0.5 * (d - cS2_over_Omega)
    kappa = math.sqrt(kappa2)
    ak = A * kappa
    common_weight = 2.0 / (kappa * d)
    power_weight = n_float * common_weight
    tau_weight = common_weight
    force_weight = 4.0 / d

    mu_i = mu[i_mu]
    sin_theta_sq = 1.0 - mu_i * mu_i
    if sin_theta_sq < 0.0:
        sin_theta_sq = 0.0
    sin_theta = math.sqrt(sin_theta_sq)
    cp = cos_phi[i_phi]
    sp = sin_phi[i_phi]
    y_component = sin_theta * sp

    k_re = 0.0
    k_im = 0.0
    dk_re = 0.0
    dk_im = 0.0
    for i_xi in range(n_xi):
        cxi = cos_xi[i_xi]
        sxi = sin_xi[i_xi]
        dot = sin_theta * (cp * (cxi - e) + sp * sqrt_one_minus_e2 * sxi)
        dot_phi = sin_theta * (-sp * (cxi - e) + cp * sqrt_one_minus_e2 * sxi)
        z = ak * dot
        z_phi = ak * dot_phi
        time_phase = n_float * xi_minus_e_sin_xi[i_xi]

        base_re = math.cos(time_phase)
        base_im = math.sin(time_phase)
        sin_q1 = math.sin(q1 * z)
        cos_q1 = math.cos(q1 * z)
        sin_q2 = math.sin(q2 * z)
        cos_q2 = math.cos(q2 * z)

        bracket_re = q1 * cos_q2 + q2 * cos_q1
        bracket_im = -q1 * sin_q2 + q2 * sin_q1
        dbracket_re = -q1 * q2 * z_phi * (sin_q1 + sin_q2)
        dbracket_im = q1 * q2 * z_phi * (cos_q1 - cos_q2)
        jac = 1.0 - e * cxi

        k_re += jac * (base_re * bracket_re - base_im * bracket_im)
        k_im += jac * (base_im * bracket_re + base_re * bracket_im)
        dk_re += jac * (base_re * dbracket_re - base_im * dbracket_im)
        dk_im += jac * (base_im * dbracket_re + base_re * dbracket_im)

    k_re /= n_xi
    k_im /= n_xi
    dk_re /= n_xi
    dk_im /= n_xi

    k_abs2 = k_re * k_re + k_im * k_im
    torque_density = k_re * dk_im - k_im * dk_re
    angle_weight = w_mu[i_mu] * (TWO_PI / n_phi)

    cuda.atomic.add(out_power, i_n, angle_weight * power_weight * k_abs2)
    cuda.atomic.add(out_tau_z, i_n, angle_weight * tau_weight * torque_density)
    cuda.atomic.add(
        out_force_y,
        i_n,
        angle_weight * force_weight * y_component * k_abs2,
    )


def _compute_quantum_terms_cuda(
    n_values: np.ndarray,
    quadrature: tuple[np.ndarray, ...],
    *,
    q1: float,
    q2: float,
    e: float,
    sqrt_one_minus_e2: float,
    A: float,
    n0: float,
    cS2_over_Omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, w_mu, cos_phi, sin_phi, cos_xi, sin_xi, xi_minus_e_sin_xi = quadrature
    d_n_values = cuda.to_device(n_values.astype(np.int32, copy=False))
    d_mu = cuda.to_device(mu)
    d_w_mu = cuda.to_device(w_mu)
    d_cos_phi = cuda.to_device(cos_phi)
    d_sin_phi = cuda.to_device(sin_phi)
    d_cos_xi = cuda.to_device(cos_xi)
    d_sin_xi = cuda.to_device(sin_xi)
    d_xi_minus_e_sin_xi = cuda.to_device(xi_minus_e_sin_xi)
    zeros = np.zeros(n_values.size, dtype=np.float64)
    d_out_power = cuda.to_device(zeros)
    d_out_tau_z = cuda.to_device(zeros)
    d_out_force_y = cuda.to_device(zeros)

    threads_per_block = 128
    total_threads = n_values.size * mu.size * cos_phi.size
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    _quantum_terms_cuda_kernel[blocks, threads_per_block](
        d_n_values,
        d_mu,
        d_w_mu,
        d_cos_phi,
        d_sin_phi,
        d_cos_xi,
        d_sin_xi,
        d_xi_minus_e_sin_xi,
        q1,
        q2,
        e,
        sqrt_one_minus_e2,
        A,
        n0,
        cS2_over_Omega,
        d_out_power,
        d_out_tau_z,
        d_out_force_y,
    )
    cuda.synchronize()
    return (
        d_out_power.copy_to_host(),
        d_out_tau_z.copy_to_host(),
        d_out_force_y.copy_to_host(),
    )


def _compute_quantum_terms_cpu(
    n_values: np.ndarray,
    quadrature: tuple[np.ndarray, ...],
    *,
    q1: float,
    q2: float,
    e: float,
    sqrt_one_minus_e2: float,
    A: float,
    n0: float,
    cS2_over_Omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _quantum_terms_cpu(
        n_values.astype(np.int32, copy=False),
        *quadrature,
        q1,
        q2,
        e,
        sqrt_one_minus_e2,
        A,
        n0,
        cS2_over_Omega,
    )


def _normalization_label(quantity: Quantity) -> str:
    if quantity == "power":
        return "P/(2*rho_bar*M^2*m_phi/sqrt(Omega))"
    if quantity == "tau_z":
        return "tau_z*tilde_Omega/(2*rho_bar*M^2*m_phi/sqrt(Omega))"
    return "F_y*sqrt(Omega)/m_phi/(2*rho_bar*M^2*m_phi/sqrt(Omega))"


def quantum_fluid_quantity(
    quantity: Quantity,
    *,
    nu: float,
    e: float,
    n0: float,
    A: float,
    cS2_over_Omega: float | None = None,
    cS_over_sqrtOmega: float = 0.0,
    n_max: int = DEFAULT_MAX_N,
    n_xi: int | None = None,
    n_mu: int = 32,
    n_phi: int = 64,
    backend: Backend = "auto",
    chunk_size: int = 64,
    rtol: float = DEFAULT_RTOL,
    atol: float = 0.0,
    tail_window: int = DEFAULT_TAIL_WINDOW,
    consecutive_windows: int = DEFAULT_CONSECUTIVE_WINDOWS,
    strict_convergence: bool = True,
    xi_per_n: int = 12,
) -> ClassicalFluidResult:
    """Compute one normalized quantum-fluid quantity."""

    if quantity not in QUANTITY_INDEX:
        raise ValueError("quantity must be 'power', 'tau_z', or 'force_y'")
    if cS2_over_Omega is None:
        if cS_over_sqrtOmega < 0.0:
            raise ValueError("cS_over_sqrtOmega = c_S/sqrt(Omega) must be non-negative")
        signed_cS2_over_Omega = cS_over_sqrtOmega * cS_over_sqrtOmega
    else:
        if cS_over_sqrtOmega != 0.0:
            raise ValueError("pass either cS2_over_Omega or cS_over_sqrtOmega, not both")
        signed_cS2_over_Omega = float(cS2_over_Omega)
    _validate_inputs(
        nu=nu,
        e=e,
        n0=n0,
        A=A,
        n_max=n_max,
        n_xi=n_xi,
        n_mu=n_mu,
        n_phi=n_phi,
        cS2_over_Omega=signed_cS2_over_Omega,
    )
    if backend not in ("auto", "cuda", "cpu"):
        raise ValueError("backend must be 'auto', 'cuda', or 'cpu'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    if tail_window < 1:
        raise ValueError("tail_window must be at least 1")
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be at least 1")
    if xi_per_n < 2:
        raise ValueError("xi_per_n must be at least 2")

    q1, q2 = mass_fractions_from_nu(nu)
    sqrt_one_minus_e2 = math.sqrt(1.0 - e * e)
    fixed_quadrature = None
    if n_xi is not None:
        fixed_quadrature = build_quadrature(n_xi, n_mu, n_phi, e)

    use_backend = backend
    if use_backend == "auto":
        use_backend = "cuda" if _cuda_available() else "cpu"
    if use_backend == "cuda" and not _cuda_available():
        raise RuntimeError("CUDA backend requested, but numba.cuda is unavailable")

    quantity_index = QUANTITY_INDEX[quantity]
    all_n: list[np.ndarray] = []
    all_terms: list[np.ndarray] = []
    total = 0.0
    converged = False
    tail_sum = math.inf
    tail_ratio = math.inf
    latest_window_sum = math.inf
    latest_chunk_sum = math.inf
    convergence_passes = 0
    max_n_xi_evaluated = 0

    compute = _compute_quantum_terms_cuda if use_backend == "cuda" else _compute_quantum_terms_cpu
    for start in range(1, n_max + 1, chunk_size):
        stop = min(n_max, start + chunk_size - 1)
        n_values = np.arange(start, stop + 1, dtype=np.int32)
        current_n_xi = n_xi
        quadrature = fixed_quadrature
        if current_n_xi is None:
            current_n_xi = recommended_n_xi(stop, xi_per_n=xi_per_n)
            quadrature = build_quadrature(current_n_xi, n_mu, n_phi, e)
        max_n_xi_evaluated = max(max_n_xi_evaluated, int(current_n_xi))

        all_quantity_terms = compute(
            n_values,
            quadrature,
            q1=q1,
            q2=q2,
            e=e,
            sqrt_one_minus_e2=sqrt_one_minus_e2,
            A=A,
            n0=n0,
            cS2_over_Omega=signed_cS2_over_Omega,
        )
        terms = all_quantity_terms[quantity_index]
        all_n.append(n_values.copy())
        all_terms.append(terms)
        latest_chunk_sum = float(np.sum(np.abs(terms)))
        total += float(np.sum(terms))

        flat_terms = np.concatenate(all_terms)
        if flat_terms.size >= tail_window:
            latest_window_sum = float(np.sum(np.abs(flat_terms[-tail_window:])))
            tail_sum = max(latest_window_sum, latest_chunk_sum)
            scale = max(abs(total), np.finfo(np.float64).tiny)
            threshold = atol + rtol * scale
            tail_ratio = tail_sum / scale
            if tail_sum <= threshold:
                convergence_passes += 1
                if convergence_passes >= consecutive_windows:
                    converged = True
                    break
            else:
                convergence_passes = 0

    n_done = np.concatenate(all_n)
    term_values = np.concatenate(all_terms)
    if not math.isfinite(tail_sum):
        tail_count = min(tail_window, term_values.size)
        latest_window_sum = float(np.sum(np.abs(term_values[-tail_count:])))
        tail_sum = max(latest_window_sum, latest_chunk_sum)
    if not math.isfinite(tail_ratio):
        scale = max(abs(float(np.sum(term_values))), np.finfo(np.float64).tiny)
        tail_ratio = tail_sum / scale

    if not converged and strict_convergence:
        raise ConvergenceError(
            f"quantum {quantity} harmonic sum did not converge before the safety cap "
            f"n_max={n_max}; last abs_tail_sum={tail_sum:.6e}, "
            f"tail_ratio={tail_ratio:.6e}, rtol={rtol:.6e}. "
            "Increase n_max/tail_window or loosen rtol if this is expected."
        )

    return ClassicalFluidResult(
        value=float(np.sum(term_values)),
        n_values=n_done,
        terms=term_values,
        backend=str(use_backend),
        converged=converged,
        tail_sum=tail_sum,
        tail_ratio=tail_ratio,
        parameters={
            "fluid": "quantum",
            "quantity": quantity,
            "normalization": _normalization_label(quantity),
            "nu": float(nu),
            "m1_over_M": q1,
            "m2_over_M": q2,
            "e": float(e),
            "n0": float(n0),
            "A": float(A),
            "A_definition": "A = a*sqrt(Omega)",
            "cS2_over_Omega": float(signed_cS2_over_Omega),
            "cS2_over_Omega_definition": "cS2_over_Omega = c_S^2/Omega, signed",
            "cS_over_sqrtOmega": (
                math.sqrt(signed_cS2_over_Omega)
                if signed_cS2_over_Omega >= 0.0
                else None
            ),
            "n_max_safety": int(n_max),
            "n_max_evaluated": int(n_done[-1]),
            "n_xi": None if n_xi is None else int(n_xi),
            "n_xi_mode": "adaptive" if n_xi is None else "fixed",
            "max_n_xi_evaluated": int(max_n_xi_evaluated),
            "n_mu": int(n_mu),
            "n_phi": int(n_phi),
            "chunk_size": int(chunk_size),
            "rtol": float(rtol),
            "atol": float(atol),
            "tail_window": int(tail_window),
            "consecutive_windows": int(consecutive_windows),
            "convergence_passes": int(convergence_passes),
            "latest_abs_window_sum": float(latest_window_sum),
            "latest_abs_chunk_sum": float(latest_chunk_sum),
        },
    )


def quantum_fluid_power(**kwargs) -> ClassicalFluidResult:
    return quantum_fluid_quantity("power", **kwargs)


def quantum_fluid_tau_z(**kwargs) -> ClassicalFluidResult:
    return quantum_fluid_quantity("tau_z", **kwargs)


def quantum_fluid_force_y(**kwargs) -> ClassicalFluidResult:
    return quantum_fluid_quantity("force_y", **kwargs)


def _write_terms_csv(path: Path, result: ClassicalFluidResult) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((result.n_values, result.terms, np.cumsum(result.terms)))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=f"n,term,cumulative_quantum_{result.parameters['quantity']}",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalized quantum-fluid observables."
    )
    parser.add_argument(
        "--quantity",
        choices=("power", "tau_z", "force_y"),
        required=True,
        help="observable to compute",
    )
    parser.add_argument("--nu", type=float, required=True, help="nu = m1*m2/M^2")
    parser.add_argument("--e", type=float, required=True, help="orbital eccentricity")
    parser.add_argument("--n0", type=float, required=True, help="n0 = m/Omega")
    parser.add_argument("--A", type=float, required=True, help="quantum A = a*sqrt(Omega)")
    parser.add_argument(
        "--cS-over-sqrtOmega",
        "--C",
        dest="cS_over_sqrtOmega",
        type=float,
        default=0.0,
        help="C = c_S/sqrt(Omega); default 0",
    )
    parser.add_argument(
        "--cS2-over-Omega",
        "--S",
        dest="cS2_over_Omega",
        type=float,
        default=None,
        help="signed S = c_S^2/Omega; do not combine with --C",
    )
    parser.add_argument("--n-max", type=int, default=DEFAULT_MAX_N)
    parser.add_argument("--n-xi", type=int, default=None)
    parser.add_argument("--n-mu", type=int, default=32)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW)
    parser.add_argument(
        "--consecutive-windows",
        type=int,
        default=DEFAULT_CONSECUTIVE_WINDOWS,
    )
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--xi-per-n", type=int, default=12)
    parser.add_argument("--save-terms", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = quantum_fluid_quantity(
        args.quantity,
        nu=args.nu,
        e=args.e,
        n0=args.n0,
        A=args.A,
        cS2_over_Omega=args.cS2_over_Omega,
        cS_over_sqrtOmega=args.cS_over_sqrtOmega,
        n_max=args.n_max,
        n_xi=args.n_xi,
        n_mu=args.n_mu,
        n_phi=args.n_phi,
        backend=args.backend,
        chunk_size=args.chunk_size,
        rtol=args.rtol,
        atol=args.atol,
        tail_window=args.tail_window,
        consecutive_windows=args.consecutive_windows,
        strict_convergence=not args.allow_unconverged,
        xi_per_n=args.xi_per_n,
    )

    print(f"quantity = {args.quantity}")
    print(f"normalized_value = {result.value:.16e}")
    print(f"normalization = {result.parameters['normalization']}")
    print(f"backend = {result.backend}")
    print(f"n_evaluated = 1..{result.n_values[-1]}")
    print(f"converged = {result.converged}")
    print(f"abs_tail_sum = {result.tail_sum:.16e}")
    print(f"tail_ratio = {result.tail_ratio:.16e}")
    print(f"rtol = {result.parameters['rtol']:.16e}")
    print(
        "grid = "
        f"n_xi:{result.parameters['n_xi_mode']} "
        f"max_n_xi:{result.parameters['max_n_xi_evaluated']} "
        f"n_mu:{result.parameters['n_mu']} "
        f"n_phi:{result.parameters['n_phi']}"
    )
    if args.save_terms is not None:
        _write_terms_csv(args.save_terms, result)
        print(f"terms_csv = {args.save_terms}")


if __name__ == "__main__":
    main()
