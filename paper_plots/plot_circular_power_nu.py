"""Circular-orbit classical power for several symmetric mass ratios.

The plotted numerical curves use the binary calculator normalization

    P_hat = P / (2 rho_bar M^2 / c_s).

For reference markers:

* black points are the equal-mass n0=0 closed form supplied by the user;
* blue points are the subsonic fixed-center single-perturber closed form,
  scaled by ``single_nu_scale**2`` to put it on the same total-mass
  normalization.
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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from classic_fluid_power import classical_fluid_power


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_circular_power_nu"),
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--nu-values", type=float, nargs="+", default=[0.25, 0.20, 0.10, 0.05])
    parser.add_argument("--n0-values", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--mach-min", type=float, default=0.08)
    parser.add_argument("--curve-fraction", type=float, default=0.95)
    parser.add_argument("--num-mach", type=int, default=22)
    parser.add_argument("--n-max", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-20)
    parser.add_argument("--n-mu", type=int, default=24)
    parser.add_argument("--n-phi", type=int, default=48)
    parser.add_argument("--xi-per-n", type=int, default=4)
    parser.add_argument("--tail-window", type=int, default=32)
    parser.add_argument("--consecutive-windows", type=int, default=2)
    parser.add_argument(
        "--single-nu-scale",
        type=float,
        default=None,
        help="nu^2 factor used for the fixed-center single-perturber analytic markers; default=min(nu-values).",
    )
    parser.add_argument("--linear-y", action="store_true")
    return parser.parse_args()


def mass_fractions_from_nu(nu: float) -> tuple[float, float]:
    if not (0.0 < nu <= 0.25):
        raise ValueError("nu must satisfy 0 < nu <= 1/4")
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * nu))
    return 0.5 * (1.0 + delta), 0.5 * (1.0 - delta)


def circular_uv_mach_limit(nu: float) -> float:
    q1, q2 = mass_fractions_from_nu(nu)
    return 1.0 / max(q1, q2)


def equal_mass_analytic_power_hat(A: float) -> float:
    """User's equal-mass circular n0=0 closed form in P_hat units."""

    component_mach = 0.5 * A
    if not (0.0 < component_mach < 1.0):
        raise ValueError("equal-mass analytic formula requires 0 < A/2 < 1")

    def equation(E: float) -> float:
        return E - component_mach * math.sin(E) - 0.5 * math.pi

    E = brentq(equation, 0.0, math.pi, xtol=1.0e-14, rtol=1.0e-14)
    bracket = (
        math.atanh(component_mach) + math.log(math.tan(0.5 * E))
    ) / (2.0 * component_mach) - 1.0
    return 2.0 * math.pi * bracket


def single_perturber_analytic_power_hat(mach: float, nu_scale: float) -> float:
    """Subsonic fixed-center single-perturber circular closed form.

    The native fixed-center result is normalized by the perturber mass squared.
    Multiplication by ``nu_scale**2`` maps it to the binary total-mass
    normalization for an EMRI reference scale.
    """

    if not (0.0 < mach < 1.0):
        raise ValueError("single-perturber analytic formula requires 0 < Mach < 1")
    native = (2.0 * math.pi / mach) * (math.atanh(mach) - mach)
    return nu_scale * nu_scale * native


def compute_numeric_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for nu in args.nu_values:
        mcrit = circular_uv_mach_limit(float(nu))
        mach_max = args.curve_fraction * mcrit
        if mach_max <= args.mach_min:
            raise ValueError(
                f"nu={nu:g}: curve_fraction*Mcrit={mach_max:.6g} is not above mach_min={args.mach_min:.6g}"
            )
        mach_values = np.linspace(args.mach_min, mach_max, args.num_mach)
        for n0 in args.n0_values:
            for idx, mach in enumerate(mach_values, start=1):
                print(
                    f"[nu={nu:g} n0={n0:g} {idx}/{len(mach_values)}] Mach={mach:.6g}",
                    flush=True,
                )
                result = classical_fluid_power(
                    nu=float(nu),
                    e=0.0,
                    n0=float(n0),
                    A=float(mach),
                    n_max=args.n_max,
                    n_mu=args.n_mu,
                    n_phi=args.n_phi,
                    backend=args.backend,
                    chunk_size=64,
                    rtol=args.rtol,
                    atol=args.atol,
                    tail_window=args.tail_window,
                    consecutive_windows=args.consecutive_windows,
                    strict_convergence=True,
                    speed_threshold_guard=True,
                    xi_per_n=args.xi_per_n,
                )
                rows.append(
                    {
                        "nu": float(nu),
                        "n0": float(n0),
                        "Mach": float(mach),
                        "Mcrit": float(mcrit),
                        "P_hat": float(result.value),
                        "converged": bool(result.converged),
                        "tail_ratio": float(result.tail_ratio),
                        "n_evaluated": int(result.n_values[-1]),
                    }
                )
    return pd.DataFrame(rows)


def build_reference_points(args: argparse.Namespace, single_nu_scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    equal_mach = np.array([0.12, 0.25, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.75, 1.88])
    single_mach = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.93])

    equal_rows = [
        {
            "kind": "equal_mass_analytic",
            "nu": 0.25,
            "n0": 0.0,
            "Mach": float(mach),
            "P_hat": equal_mass_analytic_power_hat(float(mach)),
        }
        for mach in equal_mach
    ]
    single_rows = [
        {
            "kind": "single_perturber_analytic_scaled",
            "nu_scale": float(single_nu_scale),
            "n0": 0.0,
            "Mach": float(mach),
            "P_hat": single_perturber_analytic_power_hat(float(mach), float(single_nu_scale)),
        }
        for mach in single_mach
    ]
    return pd.DataFrame(equal_rows), pd.DataFrame(single_rows)


def format_nu(nu: float) -> str:
    if abs(nu - 0.25) < 5.0e-13:
        return r"1/4"
    return f"{nu:g}"


def save_plot(
    numeric: pd.DataFrame,
    equal_points: pd.DataFrame,
    single_points: pd.DataFrame,
    args: argparse.Namespace,
    single_nu_scale: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.1))

    colors = ["#D55E00", "#009E73", "#CC79A7", "#8B1A1A", "#E69F00", "#56B4E9"]
    nu_values = [float(nu) for nu in args.nu_values]
    color_for_nu = {nu: colors[i % len(colors)] for i, nu in enumerate(nu_values)}

    for nu in nu_values:
        for n0, linestyle in [(0.0, "-"), (1.0, "--")]:
            group = numeric[(numeric["nu"] == nu) & (numeric["n0"] == n0)].sort_values("Mach")
            if group.empty:
                continue
            ax.plot(
                group["Mach"],
                group["P_hat"],
                color=color_for_nu[nu],
                lw=2.0,
                ls=linestyle,
                solid_capstyle="round",
            )

    fixed_mach = np.linspace(args.mach_min, 0.995, 320)
    fixed_power = np.array(
        [single_perturber_analytic_power_hat(float(mach), float(single_nu_scale)) for mach in fixed_mach]
    )
    ax.plot(
        fixed_mach,
        fixed_power,
        color="#0072B2",
        lw=1.7,
        ls="-",
        alpha=0.95,
        zorder=3,
    )

    for nu in nu_values:
        ax.axvline(
            circular_uv_mach_limit(nu),
            color=color_for_nu[nu],
            lw=1.15,
            ls=":",
            alpha=0.75,
        )
    ax.axvline(1.0, color="#0072B2", lw=1.15, ls=":", alpha=0.8)

    equal_mass_color = color_for_nu.get(0.25, colors[0])
    ax.scatter(
        equal_points["Mach"],
        equal_points["P_hat"],
        s=34,
        color=equal_mass_color,
        marker="o",
        zorder=5,
    )
    ax.scatter(
        single_points["Mach"],
        single_points["P_hat"],
        s=40,
        color="#0072B2",
        marker="o",
        edgecolors="white",
        linewidths=0.45,
        zorder=5,
    )
    ax.set_xlabel(r"$\mathcal{M}$", fontsize=18)
    ax.set_ylabel(r"$P/(2\bar\rho M^2/c_s)$", fontsize=18)
    if not args.linear_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=14)

    color_handles = [
        mlines.Line2D([], [], color=color_for_nu[nu], lw=2.2, label=rf"$\nu={format_nu(nu)}$")
        for nu in nu_values
    ]
    style_handles = [
        mlines.Line2D([], [], color="0.25", lw=2.0, ls="-", label=r"$n_0=0$"),
        mlines.Line2D([], [], color="0.25", lw=2.0, ls="--", label=r"$n_0=1$"),
        mlines.Line2D(
            [],
            [],
            color=equal_mass_color,
            marker="o",
            linestyle="None",
            markersize=5.5,
            label="equal-mass analytic",
        ),
        mlines.Line2D(
            [],
            [],
            color="#0072B2",
            marker="o",
            linestyle="None",
            markersize=5.8,
            label="fixed-center analytic",
        ),
    ]
    legend1 = ax.legend(handles=color_handles, loc="upper left", fontsize=12)
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, loc="lower right", fontsize=12)

    max_crit = max(circular_uv_mach_limit(nu) for nu in nu_values)
    ax.set_xlim(args.mach_min * 0.75, max_crit * 1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"circular_power_nu.{ext}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    single_nu_scale = float(args.single_nu_scale) if args.single_nu_scale is not None else min(args.nu_values)

    numeric = compute_numeric_rows(args)
    equal_points, single_points = build_reference_points(args, single_nu_scale)

    numeric_path = args.output_dir / "circular_power_nu_numeric.csv"
    refs_path = args.output_dir / "circular_power_nu_reference_points.csv"
    numeric.to_csv(numeric_path, index=False)
    pd.concat([equal_points, single_points], ignore_index=True, sort=False).to_csv(refs_path, index=False)
    save_plot(numeric, equal_points, single_points, args, single_nu_scale)

    summary = {
        "normalization": "P_hat = P/(2 rho_bar M^2/c_s)",
        "mach_definition": "Mach = a*Omega/c_s",
        "e": 0.0,
        "nu_values": [float(nu) for nu in args.nu_values],
        "n0_values": [float(n0) for n0 in args.n0_values],
        "single_perturber_marker_scaling": f"P_hat_single = nu_scale^2 * P_hat_native, nu_scale={single_nu_scale:g}",
        "fixed_center_critical_mach": 1.0,
        "uv_mach_limits": {f"nu={float(nu):g}": circular_uv_mach_limit(float(nu)) for nu in args.nu_values},
        "all_numeric_converged": bool(numeric["converged"].to_numpy().all()),
        "max_tail_ratio": float(numeric["tail_ratio"].max()),
        "numeric_csv": str(numeric_path),
        "reference_points_csv": str(refs_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"numeric csv = {numeric_path}")
    print(f"reference csv = {refs_path}")
    print(f"plot png = {args.output_dir / 'circular_power_nu.png'}")
    print(f"plot pdf = {args.output_dir / 'circular_power_nu.pdf'}")
    print(f"summary = {summary_path}")
    print(f"all_numeric_converged = {summary['all_numeric_converged']}")
    print(f"max_tail_ratio = {summary['max_tail_ratio']:.3e}")


if __name__ == "__main__":
    main()
