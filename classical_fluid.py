
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np

from classical_fluid_force_y import classical_fluid_force_y
from classical_fluid_power import (
    DEFAULT_CONSECUTIVE_WINDOWS,
    DEFAULT_MAX_N,
    DEFAULT_RTOL,
    DEFAULT_TAIL_WINDOW,
    Backend,
    ClassicalFluidResult,
    classical_fluid_power,
)
from classical_fluid_tau_z import classical_fluid_tau_z


Quantity = Literal["power", "tau_z", "force_y"]

NORMALIZATION_LABELS = {
    "power": "P/(2*rho_bar*M^2/c_s)",
    "tau_z": "tau_z*tilde_Omega/(2*rho_bar*M^2/c_s)",
    "force_y": "F_y/(2*rho_bar*M^2/c_s^2)",
}


def classical_fluid_quantity(
    quantity: Quantity,
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

    calculators = {
        "power": classical_fluid_power,
        "tau_z": classical_fluid_tau_z,
        "force_y": classical_fluid_force_y,
    }
    try:
        calculator = calculators[quantity]
    except KeyError as error:
        raise ValueError("quantity must be 'power', 'tau_z', or 'force_y'") from error
    result = calculator(
        nu=nu,
        e=e,
        n0=n0,
        A=A,
        n_max=n_max,
        n_xi=n_xi,
        n_mu=n_mu,
        n_phi=n_phi,
        backend=backend,
        chunk_size=chunk_size,
        rtol=rtol,
        atol=atol,
        tail_window=tail_window,
        consecutive_windows=consecutive_windows,
        strict_convergence=strict_convergence,
        speed_threshold_guard=speed_threshold_guard,
        xi_per_n=xi_per_n,
    )
    parameters = dict(result.parameters)
    parameters.update(
        {
            "fluid": "classical",
            "quantity": quantity,
            "normalization": NORMALIZATION_LABELS[quantity],
        }
    )
    return replace(result, parameters=parameters)


def _write_terms_csv(path: Path, result: ClassicalFluidResult) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((result.n_values, result.terms, np.cumsum(result.terms)))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=f"n,term,cumulative_classical_{result.parameters['quantity']}",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute normalized classical-fluid observables.")
    parser.add_argument("--quantity", choices=("power", "tau_z", "force_y"), required=True)
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
    parser.add_argument("--consecutive-windows", type=int, default=DEFAULT_CONSECUTIVE_WINDOWS)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--ignore-speed-threshold", action="store_true")
    parser.add_argument("--xi-per-n", type=int, default=12)
    parser.add_argument("--save-terms", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = classical_fluid_quantity(
        args.quantity,
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
    print(f"quantity = {args.quantity}")
    print(f"normalized_value = {result.value:.16e}")
    print(f"normalization = {result.parameters['normalization']}")
    print(f"backend = {result.backend}")
    print(f"n_evaluated = 1..{result.n_values[-1]}")
    print(f"converged = {result.converged}")
    print(f"abs_tail_sum = {result.tail_sum:.16e}")
    print(f"tail_ratio = {result.tail_ratio:.16e}")
    if args.save_terms is not None:
        _write_terms_csv(args.save_terms, result)
        print(f"terms_csv = {args.save_terms}")


if __name__ == "__main__":
    main()
