"""Smooth circular equal-mass quantum-fluid power curves.

For ``nu=1/4`` and ``e=0`` the full quantum power reduces to a one-dimensional
Bessel sum over even harmonics,

    P_hat = sum_even_n n/(n^2+n0^2)^(3/4)
            int dOmega J_n(beta_n sin(theta))^2,

where ``P_hat = P/(2*rho_bar*M^2*m_phi/sqrt(Omega))`` and
``beta_n = (A/2) * (n^2+n0^2)^(1/4)`` with ``A = a*sqrt(Omega)``.

The script uses this circular reduction for smooth full curves and overlays
the general quantum quadrupole approximation as sparse points.
"""

from __future__ import annotations

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
from scipy.special import jv

from quadrupole_fluxes import quantum_quadrupole_flux_normalized


OUTPUT_DIR = Path("outputs/paper_plots")
NU_EQUAL_MASS = 0.25


def circular_equal_mass_power(
    A: float,
    *,
    n0: float,
    n_max: int = 20000,
    n_mu: int = 512,
    chunk_size: int = 64,
    rtol: float = 3.0e-8,
    atol: float = 1.0e-14,
    tail_window: int = 24,
    consecutive_windows: int = 2,
) -> dict[str, float | int | bool]:
    """Return normalized equal-mass circular quantum power with convergence info."""

    if A <= 0.0:
        raise ValueError("A must be positive")
    if n0 < 0.0:
        raise ValueError("n0 must be non-negative")

    mu, w_mu = np.polynomial.legendre.leggauss(n_mu)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))

    total = 0.0
    terms: list[float] = []
    convergence_passes = 0
    converged = False
    last_n = 0
    tail_ratio = math.inf

    # Equal masses only have even harmonics in the circular orbit.
    even_values = np.arange(2, n_max + 1, 2, dtype=np.int32)
    for start in range(0, even_values.size, chunk_size):
        n_chunk = even_values[start : start + chunk_size].astype(np.float64)
        n2_plus_n02 = n_chunk * n_chunk + n0 * n0
        beta = 0.5 * A * (n2_plus_n02 ** 0.25)
        weights = n_chunk / (n2_plus_n02 ** 0.75)

        argument = beta[:, None] * sin_theta[None, :]
        angular = 2.0 * math.pi * np.sum(w_mu[None, :] * jv(n_chunk[:, None], argument) ** 2, axis=1)
        chunk_terms = weights * angular
        terms.extend(float(x) for x in chunk_terms)
        total += float(np.sum(chunk_terms))
        last_n = int(n_chunk[-1])

        if len(terms) >= tail_window:
            latest_tail = float(np.sum(np.abs(terms[-tail_window:])))
            latest_chunk = float(np.sum(np.abs(chunk_terms)))
            tail_sum = max(latest_tail, latest_chunk)
            scale = max(abs(total), np.finfo(float).tiny)
            tail_ratio = tail_sum / scale
            if tail_sum <= atol + rtol * scale:
                convergence_passes += 1
                if convergence_passes >= consecutive_windows:
                    converged = True
                    break
            else:
                convergence_passes = 0

    if not converged and last_n >= n_max:
        raise RuntimeError(
            f"circular equal-mass quantum sum did not converge for A={A:.6g}, n0={n0:g}; "
            f"last_n={last_n}, tail_ratio={tail_ratio:.3e}"
        )

    return {
        "P_hat": float(total),
        "converged": bool(converged),
        "n_evaluated": int(last_n),
        "tail_ratio": float(tail_ratio),
        "n_mu": int(n_mu),
    }


def mach_grid() -> np.ndarray:
    """A compact but denser-at-large-M grid for smooth curves."""

    low = np.linspace(0.30, 4.80, 16)
    mid = np.linspace(5.00, 20.00, 61)
    high = np.linspace(20.50, 42.00, 44)
    return np.unique(np.concatenate([low, mid, high]))


def compute_curves() -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    A_values = mach_grid()
    for n0 in (0.0, 1.0):
        for idx, A in enumerate(A_values, start=1):
            print(f"[n0={n0:g} {idx}/{len(A_values)}] M_Q={A:.6g}", flush=True)
            result = circular_equal_mass_power(float(A), n0=float(n0))
            rows.append(
                {
                    "kind": "full",
                    "n0": float(n0),
                    "M_Q": float(A),
                    "P_hat": float(result["P_hat"]),
                    "converged": bool(result["converged"]),
                    "n_evaluated": int(result["n_evaluated"]),
                    "tail_ratio": float(result["tail_ratio"]),
                    "n_mu": int(result["n_mu"]),
                }
            )
    return pd.DataFrame(rows)


def compute_quadrupole_points() -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str | None]] = []
    # Keep these sparse and in the controlled small-a*k regime.
    A_values = np.array([0.30, 0.45, 0.65, 0.90, 1.20, 1.55, 1.95, 2.40, 3.00])
    for n0 in (0.0, 1.0):
        for A in A_values:
            quad = quantum_quadrupole_flux_normalized(
                nu=NU_EQUAL_MASS,
                e=0.0,
                n0=float(n0),
                A=float(A),
                n_max=256,
                rtol=1.0e-12,
                consecutive_windows=2,
                strict_convergence=False,
                warning_ak=0.8,
            )
            rows.append(
                {
                    "kind": "quadrupole",
                    "n0": float(n0),
                    "M_Q": float(A),
                    "P_hat": float(quad.P),
                    "converged": bool(quad.converged),
                    "n_evaluated": int(quad.n_max_evaluated),
                    "tail_ratio": float(quad.tail_ratio_P),
                    "max_ak_99pct_power": float(quad.max_ak_99pct_power),
                    "warning": quad.quadrupole_warning,
                }
            )
    return pd.DataFrame(rows)


def save_plot(df_full: pd.DataFrame, df_quad: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))

    colors = {0.0: "#0072B2", 1.0: "#D55E00"}
    linestyles = {0.0: "-", 1.0: "--"}
    markers = {0.0: "o", 1.0: "s"}

    for n0 in (0.0, 1.0):
        group = df_full[df_full["n0"] == n0].sort_values("M_Q")
        ax.plot(
            group["M_Q"],
            group["P_hat"],
            color=colors[n0],
            linestyle=linestyles[n0],
            lw=2.2,
            label=rf"$n_0={int(n0)}$",
        )
        qgroup = df_quad[df_quad["n0"] == n0].sort_values("M_Q")
        ax.scatter(
            qgroup["M_Q"],
            qgroup["P_hat"],
            marker=markers[n0],
            s=42,
            facecolors="none",
            edgecolors=colors[n0],
            linewidths=1.45,
            zorder=4,
            label=rf"$n_0={int(n0)}$ (quad)",
        )

    asym_mach = np.linspace(8.0, float(df_full["M_Q"].max()), 300)
    K = 0.5 * asym_mach
    asym_x = (2.0 * math.pi * np.log(K) + math.pi * (np.euler_gamma - math.log(2.0))) / K
    ax.plot(
        asym_mach,
        asym_x,
        color="black",
        linestyle="-",
        lw=1.8,
        label=r"large-$\mathcal{M}_Q$ ($n_0=0$)",
    )

    ax.set_xlabel(r"$\mathcal{M}_Q$", fontsize=17)
    ax.set_ylabel(r"$P/(2\bar{\rho}M^2m_\phi/\sqrt{\Omega})$", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.26)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"quantum_equal_mass_power_curves.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_full = compute_curves()
    df_quad = compute_quadrupole_points()
    df_full.to_csv(OUTPUT_DIR / "quantum_equal_mass_power_curves_full.csv", index=False)
    df_quad.to_csv(OUTPUT_DIR / "quantum_equal_mass_power_curves_quadrupole.csv", index=False)
    save_plot(df_full, df_quad)

    summary = {
        "normalization": "P/(2*rho_bar*M^2*m_phi/sqrt(Omega))",
        "mach_definition": "M_Q = A = a*sqrt(Omega)",
        "nu": NU_EQUAL_MASS,
        "e": 0.0,
        "n0_values": [0.0, 1.0],
        "full_curve_method": "equal-mass circular Bessel reduction of the full quantum calculator",
        "full_all_converged": bool(df_full["converged"].to_numpy().all()),
        "full_max_tail_ratio": float(df_full["tail_ratio"].max()),
        "full_max_n_evaluated": int(df_full["n_evaluated"].max()),
        "outputs": {
            "png": str(OUTPUT_DIR / "quantum_equal_mass_power_curves.png"),
            "pdf": str(OUTPUT_DIR / "quantum_equal_mass_power_curves.pdf"),
            "full_csv": str(OUTPUT_DIR / "quantum_equal_mass_power_curves_full.csv"),
            "quadrupole_csv": str(OUTPUT_DIR / "quantum_equal_mass_power_curves_quadrupole.csv"),
        },
    }
    (OUTPUT_DIR / "quantum_equal_mass_power_curves_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"plot png = {OUTPUT_DIR / 'quantum_equal_mass_power_curves.png'}")
    print(f"plot pdf = {OUTPUT_DIR / 'quantum_equal_mass_power_curves.pdf'}")
    print(f"full csv = {OUTPUT_DIR / 'quantum_equal_mass_power_curves_full.csv'}")
    print(f"quad csv = {OUTPUT_DIR / 'quantum_equal_mass_power_curves_quadrupole.csv'}")
    print(f"summary = {OUTPUT_DIR / 'quantum_equal_mass_power_curves_summary.json'}")
    print(f"all converged = {summary['full_all_converged']}")
    print(f"max tail ratio = {summary['full_max_tail_ratio']:.3e}")
    print(f"max n evaluated = {summary['full_max_n_evaluated']}")


if __name__ == "__main__":
    main()
