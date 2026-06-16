"""Classical-fluid eccentricity-scan flux plot.

The figure has three panels for normalized P, tau_z, and -F_y versus Mach
number.  The line color labels eccentricity.
Solid lines are n0=0 and dashed lines are n0=1.

The point-source UV threshold is

    Mcrit(e, nu) = sqrt((1 - e)/(1 + e)) / max(m1/M, m2/M).

Curves are evaluated only below this threshold; same-color vertical dotted
lines mark the threshold locations.
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

from classic_fluid_force_y import classical_fluid_force_y
from classic_fluid_power import classical_fluid_power
from classic_fluid_tau_z import classical_fluid_tau_z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classical_ecc_fluxes"),
    )
    parser.add_argument("--figure-stem", type=str, default="classical_ecc_fluxes")
    parser.add_argument(
        "--report-title",
        type=str,
        default="Classical-Fluid Eccentricity Scan",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--eccentricities", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.8])
    parser.add_argument("--nu", type=float, default=0.25)
    parser.add_argument("--mach-min", type=float, default=0.05)
    parser.add_argument("--curve-fraction", type=float, default=0.92)
    parser.add_argument("--num-mach", type=int, default=18)
    parser.add_argument("--n-max", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--n-mu", type=int, default=24)
    parser.add_argument("--n-phi", type=int, default=48)
    parser.add_argument("--xi-per-n", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1.0e-20)
    parser.add_argument("--linear-y", action="store_true")
    return parser.parse_args()


def mass_fractions_from_nu_local(nu: float) -> tuple[float, float]:
    if not (0.0 < nu <= 0.25):
        raise ValueError("nu must satisfy 0 < nu <= 1/4")
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * nu))
    return 0.5 * (1.0 + delta), 0.5 * (1.0 - delta)


def uv_mach_limit(e: float, nu: float) -> float:
    if not (0.0 <= e < 1.0):
        raise ValueError("e must satisfy 0 <= e < 1")
    q1, q2 = mass_fractions_from_nu_local(nu)
    return math.sqrt((1.0 - e) / (1.0 + e)) / max(q1, q2)


def compute_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    for e in args.eccentricities:
        mcrit = uv_mach_limit(float(e), args.nu)
        mach_max = args.curve_fraction * mcrit
        if mach_max <= args.mach_min:
            raise ValueError(
                f"curve_fraction*Mcrit={mach_max:.6g} is not above mach_min={args.mach_min:.6g}"
            )
        mach_values = np.linspace(args.mach_min, mach_max, args.num_mach)
        for n0 in (0.0, 1.0):
            for idx, mach in enumerate(mach_values, start=1):
                print(
                    f"[e={e:g} n0={n0:g} {idx}/{len(mach_values)}] Mach={mach:.6g}",
                    flush=True,
                )
                common = dict(
                    nu=args.nu,
                    e=float(e),
                    n0=float(n0),
                    A=float(mach),
                    n_max=args.n_max,
                    n_mu=args.n_mu,
                    n_phi=args.n_phi,
                    backend=args.backend,
                    chunk_size=64,
                    rtol=args.rtol,
                    atol=args.atol,
                    tail_window=32,
                    consecutive_windows=2,
                    strict_convergence=True,
                    speed_threshold_guard=True,
                    xi_per_n=args.xi_per_n,
                )
                p = classical_fluid_power(**common)
                tau = classical_fluid_tau_z(**common)
                if abs(args.nu - 0.25) < 1.0e-14:
                    # Equal masses have no net orbit-averaged y-force by symmetry.
                    fy_value = 0.0
                    minus_fy = np.nan
                    fy_converged = True
                    fy_n = 0
                    fy_tail = 0.0
                else:
                    fy = classical_fluid_force_y(**common)
                    fy_value = float(fy.value)
                    minus_fy = float(-fy.value)
                    fy_converged = bool(fy.converged)
                    fy_n = int(fy.n_values[-1])
                    fy_tail = float(fy.tail_ratio)
                rows.append(
                    {
                        "e": float(e),
                        "n0": float(n0),
                        "Mach": float(mach),
                        "Mcrit": float(mcrit),
                        "P_hat": float(p.value),
                        "tau_hat": float(tau.value),
                        "F_y_hat": fy_value,
                        "minus_F_y_hat": minus_fy,
                        "P_converged": bool(p.converged),
                        "tau_converged": bool(tau.converged),
                        "Fy_converged": fy_converged,
                        "P_n": int(p.n_values[-1]),
                        "tau_n": int(tau.n_values[-1]),
                        "Fy_n": fy_n,
                        "P_tail": float(p.tail_ratio),
                        "tau_tail": float(tau.tail_ratio),
                        "Fy_tail": fy_tail,
                    }
                )
    return pd.DataFrame(rows)


def monotonic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for e, group in df.groupby("e"):
        g0 = group[group["n0"] == 0.0].set_index("Mach").sort_index()
        g1 = group[group["n0"] == 1.0].set_index("Mach").sort_index()
        for quantity, col in [
            ("P", "P_hat"),
            ("tau_z", "tau_hat"),
            ("-F_y", "minus_F_y_hat"),
        ]:
            joined = pd.DataFrame({"n0_0": g0[col], "n0_1": g1[col]}).dropna()
            if joined.empty:
                continue
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
        ("P_hat", r"$P/(2\bar\rho M^2/c_s)$"),
        ("tau_hat", r"$\tau_z\Omega/(2\bar\rho M^2/c_s)$"),
        ("minus_F_y_hat", r"$-F_y/(2\bar\rho M^2/c_s^2)$"),
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    e_values = [float(e) for e in args.eccentricities]
    colors = {e: color_cycle[i % len(color_cycle)] for i, e in enumerate(e_values)}

    for ax, (col, ylabel) in zip(axes, quantities):
        for e in e_values:
            color = colors[e]
            for n0, linestyle in [(0.0, "-"), (1.0, "--")]:
                group = df[(df["e"] == e) & (df["n0"] == n0)].sort_values("Mach")
                if group.empty:
                    continue
                y = group[col]
                if y.isna().all():
                    continue
                label = rf"$e={e:g},\,n_0={int(n0)}$"
                ax.plot(
                    group["Mach"],
                    y,
                    color=color,
                    lw=2.0,
                    ls=linestyle,
                    label=label,
                )
            ax.axvline(
                uv_mach_limit(e, args.nu),
                color=color,
                lw=1.2,
                ls=":",
                alpha=0.75,
            )

        ax.set_xlabel(r"$\mathcal{M}$", fontsize=14)
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
        elif not args.linear_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=11)

    max_crit = max(uv_mach_limit(e, args.nu) for e in e_values)
    axes[0].set_xlim(args.mach_min * 0.9, max_crit * 1.05)
    axes[0].legend(fontsize=6.6, loc="lower right", ncol=1)
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
    mono_path = args.output_dir / "n0_monotonic_check.csv"
    df.to_csv(data_path, index=False)
    mono.to_csv(mono_path, index=False)
    save_plot(df, args, args.output_dir)

    full_convergence_cols = ["P_converged", "tau_converged"]
    full_converged = bool(df[full_convergence_cols].to_numpy().all())
    fy_rows = df[~df["F_y_hat"].isna()]
    if not fy_rows.empty:
        full_converged = full_converged and bool(fy_rows["Fy_converged"].to_numpy().all())

    summary = {
        "nu": args.nu,
        "eccentricities": [float(e) for e in args.eccentricities],
        "n0_values": [0, 1],
        "uv_mach_limits": {
            f"e={float(e):g}": uv_mach_limit(float(e), args.nu)
            for e in args.eccentricities
        },
        "curve_fraction_of_Mcrit": args.curve_fraction,
        "normalization": {
            "P_hat": "P/(2*rho_bar*M^2/c_s)",
            "tau_hat": "tau_z*Omega/(2*rho_bar*M^2/c_s)",
            "F_y_hat": "F_y/(2*rho_bar*M^2/c_s^2)",
        },
        "convergence": {
            "all_converged": full_converged,
            "max_n": int(df[["P_n", "tau_n", "Fy_n"]].to_numpy().max()),
            "max_tail": float(np.nanmax(df[["P_tail", "tau_tail", "Fy_tail"]].to_numpy())),
            "linear_y": bool(args.linear_y),
        },
        "n0_monotonic_check": mono.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        f"# {args.report_title}",
        "",
        "Files:",
        "",
        f"- `{args.figure_stem}.png/pdf`",
        f"- `{args.figure_stem}_data.csv`",
        "- `n0_monotonic_check.csv`",
        "- `summary.json`",
        "",
        "Summary:",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"figure = {args.output_dir / f'{args.figure_stem}.png'}")
    print(f"data = {data_path}")
    print(f"monotonic_check = {mono_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
