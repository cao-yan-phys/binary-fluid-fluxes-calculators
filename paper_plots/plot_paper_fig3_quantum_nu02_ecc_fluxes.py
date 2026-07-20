"""Paper Fig. 3 quantum-fluid counterpart for eccentric binaries with nu=0.2.

Three panels show the normalized quantum-fluid power, torque, and y-force
component versus

    M_Q = A = a*sqrt(Omega).

The line color labels eccentricity, solid lines are ``n0=0``, and dashed
lines are ``n0=1``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classic_fluid_power import build_quadrature, mass_fractions_from_nu, recommended_n_xi
from quantum_fluid import (
    _compute_quantum_terms_cpu,
    _compute_quantum_terms_cuda,
    _cuda_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_plots"),
    )
    parser.add_argument("--figure-stem", type=str, default="paper_fig3_quantum_nu02_ecc_fluxes")
    parser.add_argument(
        "--report-title",
        type=str,
        default="Paper Fig. 3: Quantum nu=0.2 Eccentricity Scan",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--eccentricities", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.8])
    parser.add_argument("--nu", type=float, default=0.20)
    parser.add_argument("--mach-min", type=float, default=0.30)
    parser.add_argument("--mach-max", type=float, default=15.0)
    parser.add_argument("--num-mach", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=3.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-14)
    parser.add_argument("--n-mu", type=int, default=20)
    parser.add_argument("--n-phi", type=int, default=40)
    parser.add_argument("--xi-per-n", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--tail-window", type=int, default=32)
    parser.add_argument("--consecutive-windows", type=int, default=2)
    parser.add_argument("--linear-y", dest="linear_y", action="store_true", default=True)
    parser.add_argument("--log-y", dest="linear_y", action="store_false")
    return parser.parse_args()


def mach_grid(mach_min: float, mach_max: float, num_mach: int) -> np.ndarray:
    """Return a compact grid with a little more support above M_Q~5."""

    if num_mach < 8:
        raise ValueError("num_mach must be at least 8")
    split = min(5.0, mach_max)
    n_low = max(5, int(round(0.42 * num_mach)))
    n_high = num_mach - n_low + 1
    low = np.linspace(mach_min, split, n_low)
    high = np.linspace(split, mach_max, n_high)
    return np.unique(np.concatenate([low, high]))


def quantum_power_tau_force(
    *,
    nu: float,
    e: float,
    n0: float,
    A: float,
    n_max: int,
    n_mu: int,
    n_phi: int,
    backend: str,
    chunk_size: int,
    rtol: float,
    atol: float,
    tail_window: int,
    consecutive_windows: int,
    xi_per_n: int,
) -> dict[str, float | int | bool | str]:
    """Compute quantum P, tau_z, and F_y on the same harmonic/angular grid."""

    q1, q2 = mass_fractions_from_nu(nu)
    sqrt_one_minus_e2 = math.sqrt(1.0 - e * e)

    use_backend = backend
    if use_backend == "auto":
        use_backend = "cuda" if _cuda_available() else "cpu"
    if use_backend == "cuda" and not _cuda_available():
        raise RuntimeError("CUDA backend requested, but numba.cuda is unavailable")
    compute = _compute_quantum_terms_cuda if use_backend == "cuda" else _compute_quantum_terms_cpu

    all_p: list[np.ndarray] = []
    all_tau: list[np.ndarray] = []
    all_force: list[np.ndarray] = []
    total_p = 0.0
    total_tau = 0.0
    total_force = 0.0
    converged = False
    convergence_passes = 0
    max_n_xi = 0
    latest_tail_ratio_p = math.inf
    latest_tail_ratio_tau = math.inf
    n_done = 0

    for start in range(1, n_max + 1, chunk_size):
        stop = min(n_max, start + chunk_size - 1)
        n_values = np.arange(start, stop + 1, dtype=np.int32)
        current_n_xi = recommended_n_xi(stop, xi_per_n=xi_per_n)
        max_n_xi = max(max_n_xi, int(current_n_xi))
        quadrature = build_quadrature(current_n_xi, n_mu, n_phi, e)
        p_terms, tau_terms, force_terms = compute(
            n_values,
            quadrature,
            q1=q1,
            q2=q2,
            e=e,
            sqrt_one_minus_e2=sqrt_one_minus_e2,
            A=A,
            n0=n0,
            cS2_over_Omega=0.0,
        )
        all_p.append(p_terms)
        all_tau.append(tau_terms)
        all_force.append(force_terms)
        total_p += float(np.sum(p_terms))
        total_tau += float(np.sum(tau_terms))
        total_force += float(np.sum(force_terms))
        n_done = int(stop)

        p_flat = np.concatenate(all_p)
        tau_flat = np.concatenate(all_tau)
        if p_flat.size >= tail_window:
            p_tail_sum = max(
                float(np.sum(np.abs(p_flat[-tail_window:]))),
                float(np.sum(np.abs(p_terms))),
            )
            tau_tail_sum = max(
                float(np.sum(np.abs(tau_flat[-tail_window:]))),
                float(np.sum(np.abs(tau_terms))),
            )
            p_scale = max(abs(total_p), np.finfo(float).tiny)
            tau_scale = max(abs(total_tau), np.finfo(float).tiny)
            latest_tail_ratio_p = p_tail_sum / p_scale
            latest_tail_ratio_tau = tau_tail_sum / tau_scale
            p_ok = p_tail_sum <= atol + rtol * p_scale
            tau_ok = tau_tail_sum <= atol + rtol * tau_scale
            if p_ok and tau_ok:
                convergence_passes += 1
                if convergence_passes >= consecutive_windows:
                    converged = True
                    break
            else:
                convergence_passes = 0

    if not converged:
        raise RuntimeError(
            f"quantum P/tau did not converge for e={e:g}, n0={n0:g}, M_Q={A:g}; "
            f"n_done={n_done}, tail_P={latest_tail_ratio_p:.3e}, "
            f"tail_tau={latest_tail_ratio_tau:.3e}"
        )

    return {
        "P_hat": float(total_p),
        "tau_hat": float(total_tau),
        "F_y_hat": float(total_force),
        "backend": str(use_backend),
        "converged": bool(converged),
        "n_evaluated": int(n_done),
        "max_n_xi": int(max_n_xi),
        "P_tail": float(latest_tail_ratio_p),
        "tau_tail": float(latest_tail_ratio_tau),
        "F_y_tail": float(
            np.sum(np.abs(np.concatenate(all_force)[-min(tail_window, sum(x.size for x in all_force)) :]))
            / max(abs(total_force), np.finfo(float).tiny)
        ),
    }


def compute_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    mach_values = mach_grid(args.mach_min, args.mach_max, args.num_mach)
    for e in args.eccentricities:
        for n0 in (0.0, 1.0):
            for idx, mach in enumerate(mach_values, start=1):
                print(
                    f"[e={e:g} n0={n0:g} {idx}/{len(mach_values)}] M_Q={mach:.6g}",
                    flush=True,
                )
                result = quantum_power_tau_force(
                    nu=args.nu,
                    e=float(e),
                    n0=float(n0),
                    A=float(mach),
                    n_max=args.n_max,
                    n_mu=args.n_mu,
                    n_phi=args.n_phi,
                    backend=args.backend,
                    chunk_size=args.chunk_size,
                    rtol=args.rtol,
                    atol=args.atol,
                    tail_window=args.tail_window,
                    consecutive_windows=args.consecutive_windows,
                    xi_per_n=args.xi_per_n,
                )
                force_zero_by_symmetry = abs(args.nu - 0.25) < 1.0e-14 or abs(float(e)) < 1.0e-14
                f_y_hat = 0.0 if force_zero_by_symmetry else float(result["F_y_hat"])
                minus_f_y_hat = np.nan if force_zero_by_symmetry else -f_y_hat
                rows.append(
                    {
                        "e": float(e),
                        "n0": float(n0),
                        "M_Q": float(mach),
                        "P_hat": float(result["P_hat"]),
                        "tau_hat": float(result["tau_hat"]),
                        "F_y_hat": f_y_hat,
                        "minus_F_y_hat": minus_f_y_hat,
                        "converged": bool(result["converged"]),
                        "P_tail": float(result["P_tail"]),
                        "tau_tail": float(result["tau_tail"]),
                        "F_y_tail": float(result["F_y_tail"]),
                        "n_evaluated": int(result["n_evaluated"]),
                        "max_n_xi": int(result["max_n_xi"]),
                        "backend": str(result["backend"]),
                    }
                )
    return pd.DataFrame(rows)


def monotonic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for e, group in df.groupby("e"):
        g0 = group[group["n0"] == 0.0].set_index("M_Q").sort_index()
        g1 = group[group["n0"] == 1.0].set_index("M_Q").sort_index()
        for quantity, col in [("P", "P_hat"), ("tau_z", "tau_hat")]:
            joined = pd.DataFrame({"n0_0": g0[col], "n0_1": g1[col]}).dropna()
            ratio = joined["n0_1"] / joined["n0_0"]
            rows.append(
                {
                    "e": float(e),
                    "quantity": quantity,
                    "all_n0_1_ge_n0_0": bool((ratio >= 1.0).all()),
                    "min_ratio_n0_1_over_n0_0": float(ratio.min()),
                    "max_ratio_n0_1_over_n0_0": float(ratio.max()),
                }
            )
    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True)
    quantities = [
        ("P_hat", r"$P/(2\bar\rho M^2m_\phi/\sqrt{\Omega})$"),
        ("tau_hat", r"$\tau_z\tilde\Omega/(2\bar\rho M^2m_\phi/\sqrt{\Omega})$"),
        (
            "minus_F_y_hat",
            r"$-F_y\sqrt{\Omega}/m_\phi\,/(2\bar\rho M^2m_\phi/\sqrt{\Omega})$",
        ),
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    e_values = [float(e) for e in args.eccentricities]
    colors = {e: color_cycle[i % len(color_cycle)] for i, e in enumerate(e_values)}

    for ax, (col, ylabel) in zip(axes, quantities):
        for e in e_values:
            color = colors[e]
            for n0, linestyle in [(0.0, "-"), (1.0, "--")]:
                group = df[(df["e"] == e) & (df["n0"] == n0)].sort_values("M_Q")
                if group.empty:
                    continue
                y = group[col]
                if y.isna().all():
                    continue
                ax.plot(
                    group["M_Q"],
                    y,
                    color=color,
                    lw=2.0,
                    ls=linestyle,
                    label=rf"$e={e:g},\,n_0={int(n0)}$",
                )

        ax.set_xlabel(r"$\mathcal{M}_Q$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if col == "minus_F_y_hat" and df[col].dropna().empty:
            ax.set_ylim(-1.0, 1.0)
            ax.text(
                0.5,
                0.5,
                r"$F_y=0$ for $m_1=m_2$",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
            )
        elif col != "minus_F_y_hat" and not args.linear_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=11)

    axes[0].set_xlim(args.mach_min * 0.9, args.mach_max * 1.02)
    axes[0].legend(fontsize=6.6, loc="best", ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{args.figure_stem}.{ext}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = compute_rows(args)
    mono = monotonic_summary(df)
    data_path = args.output_dir / f"{args.figure_stem}_data.csv"
    mono_path = args.output_dir / f"{args.figure_stem}_n0_monotonic_check.csv"
    df.to_csv(data_path, index=False)
    mono.to_csv(mono_path, index=False)
    save_plot(df, args, args.output_dir)

    summary = {
        "nu": args.nu,
        "eccentricities": [float(e) for e in args.eccentricities],
        "n0_values": [0, 1],
        "mach_definition": "M_Q = a*sqrt(Omega)",
        "mach_range": [args.mach_min, args.mach_max],
        "normalization": {
            "P_hat": "P/(2*rho_bar*M^2*m_phi/sqrt(Omega))",
            "tau_hat": "tau_z*tildeOmega/(2*rho_bar*M^2*m_phi/sqrt(Omega))",
            "F_y_hat": "F_y*sqrt(Omega)/m_phi/(2*rho_bar*M^2*m_phi/sqrt(Omega))",
        },
        "convergence": {
            "all_converged": bool(df["converged"].to_numpy().all()),
            "max_n": int(df["n_evaluated"].max()),
            "max_n_xi": int(df["max_n_xi"].max()),
            "max_tail_P_tau": float(np.nanmax(df[["P_tail", "tau_tail"]].to_numpy())),
            "linear_y": bool(args.linear_y),
        },
        "n0_monotonic_check": mono.to_dict(orient="records"),
    }
    summary_path = args.output_dir / f"{args.figure_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        f"# {args.report_title}",
        "",
        "Files:",
        "",
        f"- `{args.figure_stem}.png/pdf`",
        f"- `{args.figure_stem}_data.csv`",
        f"- `{args.figure_stem}_n0_monotonic_check.csv`",
        f"- `{args.figure_stem}_summary.json`",
        "",
        "Summary:",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    (args.output_dir / f"{args.figure_stem}_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(f"figure = {args.output_dir / f'{args.figure_stem}.png'}")
    print(f"data = {data_path}")
    print(f"monotonic_check = {mono_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
