"""Calculator for the classical-fluid normalized y force.

The returned value is

    F_y / (2 * rho_bar * M**2 / c_s**2)

using the implemented normalized harmonic expression:

    sum_{n>=1} int dOmega |K_n|^2 sin(theta) sin(phi).

The CUDA backend mirrors classic_fluid_power.py: each thread handles one
`(n, mu, phi)` point and performs the oscillatory `xi` integral locally.
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
    DivergenceError,
    build_quadrature,
    mass_fractions_from_nu,
    recommended_n_xi,
    speed_threshold_ratio,
)


Backend = Literal["auto", "cuda", "cpu"]


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
) -> None:
    mass_fractions_from_nu(nu)
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    if n0 < 0.0:
        raise ValueError("n0 must be non-negative")
    if A < 0.0:
        raise ValueError("A = a*Omega must be non-negative")
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    if n_xi is not None and n_xi < 8:
        raise ValueError("n_xi must be at least 8")
    if n_mu < 2:
        raise ValueError("n_mu must be at least 2")
    if n_phi < 4:
        raise ValueError("n_phi must be at least 4")


@njit(parallel=True, fastmath=True)
def _force_y_terms_cpu(
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
) -> np.ndarray:
    out = np.zeros(n_values.size, dtype=np.float64)
    n_phi = cos_phi.size
    n_xi = cos_xi.size
    phi_weight = TWO_PI / n_phi

    for i_n in prange(n_values.size):
        n_float = float(n_values[i_n])
        ratio = n0 / n_float
        dispersion = math.sqrt(1.0 + ratio * ratio)
        ak = A * n_float * dispersion
        harmonic_sum = 0.0

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

                for i_xi in range(n_xi):
                    cxi = cos_xi[i_xi]
                    sxi = sin_xi[i_xi]
                    dot = sin_theta * (
                        cp * (cxi - e) + sp * sqrt_one_minus_e2 * sxi
                    )
                    z = ak * dot
                    time_phase = n_float * xi_minus_e_sin_xi[i_xi]

                    base_re = math.cos(time_phase)
                    base_im = math.sin(time_phase)
                    bracket_re = q1 * math.cos(q2 * z) + q2 * math.cos(q1 * z)
                    bracket_im = -q1 * math.sin(q2 * z) + q2 * math.sin(q1 * z)
                    jac = 1.0 - e * cxi

                    k_re += jac * (base_re * bracket_re - base_im * bracket_im)
                    k_im += jac * (base_im * bracket_re + base_re * bracket_im)

                k_re /= n_xi
                k_im /= n_xi
                harmonic_sum += (
                    mu_weight
                    * phi_weight
                    * y_component
                    * (k_re * k_re + k_im * k_im)
                )

        out[i_n] = harmonic_sum

    return out


@cuda.jit
def _force_y_terms_cuda_kernel(
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
    out,
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
    ratio = n0 / n_float
    dispersion = math.sqrt(1.0 + ratio * ratio)
    ak = A * n_float * dispersion

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
    for i_xi in range(n_xi):
        cxi = cos_xi[i_xi]
        sxi = sin_xi[i_xi]
        dot = sin_theta * (cp * (cxi - e) + sp * sqrt_one_minus_e2 * sxi)
        z = ak * dot
        time_phase = n_float * xi_minus_e_sin_xi[i_xi]

        base_re = math.cos(time_phase)
        base_im = math.sin(time_phase)
        bracket_re = q1 * math.cos(q2 * z) + q2 * math.cos(q1 * z)
        bracket_im = -q1 * math.sin(q2 * z) + q2 * math.sin(q1 * z)
        jac = 1.0 - e * cxi

        k_re += jac * (base_re * bracket_re - base_im * bracket_im)
        k_im += jac * (base_im * bracket_re + base_re * bracket_im)

    k_re /= n_xi
    k_im /= n_xi
    phi_weight = TWO_PI / n_phi
    contribution = w_mu[i_mu] * phi_weight * y_component * (
        k_re * k_re + k_im * k_im
    )
    cuda.atomic.add(out, i_n, contribution)


def _compute_force_y_terms_cuda(
    n_values: np.ndarray,
    quadrature: tuple[np.ndarray, ...],
    *,
    q1: float,
    q2: float,
    e: float,
    sqrt_one_minus_e2: float,
    A: float,
    n0: float,
) -> np.ndarray:
    mu, w_mu, cos_phi, sin_phi, cos_xi, sin_xi, xi_minus_e_sin_xi = quadrature
    d_n_values = cuda.to_device(n_values.astype(np.int32, copy=False))
    d_mu = cuda.to_device(mu)
    d_w_mu = cuda.to_device(w_mu)
    d_cos_phi = cuda.to_device(cos_phi)
    d_sin_phi = cuda.to_device(sin_phi)
    d_cos_xi = cuda.to_device(cos_xi)
    d_sin_xi = cuda.to_device(sin_xi)
    d_xi_minus_e_sin_xi = cuda.to_device(xi_minus_e_sin_xi)
    d_out = cuda.to_device(np.zeros(n_values.size, dtype=np.float64))

    threads_per_block = 128
    total_threads = n_values.size * mu.size * cos_phi.size
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    _force_y_terms_cuda_kernel[blocks, threads_per_block](
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
        d_out,
    )
    cuda.synchronize()
    return d_out.copy_to_host()


def _compute_force_y_terms_cpu(
    n_values: np.ndarray,
    quadrature: tuple[np.ndarray, ...],
    *,
    q1: float,
    q2: float,
    e: float,
    sqrt_one_minus_e2: float,
    A: float,
    n0: float,
) -> np.ndarray:
    return _force_y_terms_cpu(
        n_values.astype(np.int32, copy=False),
        *quadrature,
        q1,
        q2,
        e,
        sqrt_one_minus_e2,
        A,
        n0,
    )


def classical_fluid_force_y(
    *,
    nu: float,
    e: float,
    n0: float,
    A: float,
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
    speed_threshold_guard: bool = True,
    xi_per_n: int = 12,
) -> ClassicalFluidResult:
    """Compute F_y / (2*rho_bar*M^2/c_s^2)."""

    _validate_inputs(
        nu=nu,
        e=e,
        n0=n0,
        A=A,
        n_max=n_max,
        n_xi=n_xi,
        n_mu=n_mu,
        n_phi=n_phi,
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

    threshold_ratio = speed_threshold_ratio(nu, e, A)
    if speed_threshold_guard and threshold_ratio >= 1.0:
        raise DivergenceError(
            "parameters satisfy the large-n body-speed divergence criterion: "
            "max(m1/M,m2/M) * A * sqrt((1+e)/(1-e)) "
            f"= {threshold_ratio:.12g} >= 1."
        )

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

    compute = (
        _compute_force_y_terms_cuda if use_backend == "cuda" else _compute_force_y_terms_cpu
    )
    for start in range(1, n_max + 1, chunk_size):
        stop = min(n_max, start + chunk_size - 1)
        n_values = np.arange(start, stop + 1, dtype=np.int32)
        current_n_xi = n_xi
        quadrature = fixed_quadrature
        if current_n_xi is None:
            current_n_xi = recommended_n_xi(stop, xi_per_n=xi_per_n)
            quadrature = build_quadrature(current_n_xi, n_mu, n_phi, e)
        max_n_xi_evaluated = max(max_n_xi_evaluated, int(current_n_xi))

        terms = compute(
            n_values,
            quadrature,
            q1=q1,
            q2=q2,
            e=e,
            sqrt_one_minus_e2=sqrt_one_minus_e2,
            A=A,
            n0=n0,
        )
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
            "F_y harmonic sum did not converge before the safety cap "
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
            "quantity": "force_y_normalized",
            "normalization": "F_y/(2*rho_bar*M^2/c_s^2)",
            "nu": float(nu),
            "m1_over_M": q1,
            "m2_over_M": q2,
            "e": float(e),
            "n0": float(n0),
            "A": float(A),
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
            "speed_threshold_ratio": float(threshold_ratio),
            "speed_threshold_guard": bool(speed_threshold_guard),
        },
    )


def _write_terms_csv(path: Path, result: ClassicalFluidResult) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((result.n_values, result.terms, np.cumsum(result.terms)))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="n,term,cumulative_normalized_force_y",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute F_y/(2*rho_bar*M^2/c_s^2) for the classical-fluid formulas."
    )
    parser.add_argument("--nu", type=float, required=True, help="nu = m1*m2/M^2")
    parser.add_argument("--e", type=float, required=True, help="orbital eccentricity")
    parser.add_argument("--n0", type=float, required=True, help="n0 = m/Omega")
    parser.add_argument("--A", type=float, required=True, help="A = a*Omega")
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
    parser.add_argument("--ignore-speed-threshold", action="store_true")
    parser.add_argument("--xi-per-n", type=int, default=12)
    parser.add_argument("--save-terms", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = classical_fluid_force_y(
        nu=args.nu,
        e=args.e,
        n0=args.n0,
        A=args.A,
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
        speed_threshold_guard=not args.ignore_speed_threshold,
        xi_per_n=args.xi_per_n,
    )

    print(f"normalized_force_y = {result.value:.16e}")
    print(f"backend = {result.backend}")
    print(f"n_evaluated = 1..{result.n_values[-1]}")
    print(f"converged = {result.converged}")
    print(f"abs_tail_sum = {result.tail_sum:.16e}")
    print(f"tail_ratio = {result.tail_ratio:.16e}")
    print(f"rtol = {result.parameters['rtol']:.16e}")
    print(f"speed_threshold_ratio = {result.parameters['speed_threshold_ratio']:.16e}")
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
