from __future__ import annotations

import argparse
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

from paper_plots.plot_paper_fig3_quantum_nu02_ecc_fluxes import (
    mach_grid,
    quantum_power_tau_force,
)


DATA_PATH = Path("outputs/paper_plots/paper_fig3_quantum_nu02_ecc_edot_data.csv")
OUTPUT_PATH = Path("outputs/paper_plots/paper_fig3_quantum_nu02_ecc_edot.pdf")
E_VALUES = (0.2, 0.4, 0.8)
NU = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def edot_hat(*, e: float, A: float, power_hat: float, tau_hat: float) -> float:
    return -A * (1.0 - e * e) / e * (
        power_hat - tau_hat / math.sqrt(1.0 - e * e)
    )


def compute_data(backend: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    mach_values = mach_grid(0.30, 15.0, 20)
    for e in E_VALUES:
        for n0 in (0.0, 1.0):
            for index, A in enumerate(mach_values, start=1):
                print(f"[e={e:g}, n0={n0:g}, {index}/{len(mach_values)}] M_Q={A:.6g}")
                result = quantum_power_tau_force(
                    nu=NU,
                    e=e,
                    n0=n0,
                    A=float(A),
                    n_max=4096,
                    n_mu=20,
                    n_phi=40,
                    backend=backend,
                    chunk_size=64,
                    rtol=3.0e-6,
                    atol=1.0e-14,
                    tail_window=32,
                    consecutive_windows=2,
                    xi_per_n=4,
                )
                power_hat = float(result["P_hat"])
                tau_hat = float(result["tau_hat"])
                rows.append(
                    {
                        "e": e,
                        "n0": n0,
                        "M_Q": float(A),
                        "P_hat": power_hat,
                        "tau_hat": tau_hat,
                        "edot_hat": edot_hat(
                            e=e,
                            A=float(A),
                            power_hat=power_hat,
                            tau_hat=tau_hat,
                        ),
                        "converged": bool(result["converged"]),
                        "n_evaluated": int(result["n_evaluated"]),
                        "max_n_xi": int(result["max_n_xi"]),
                        "P_tail": float(result["P_tail"]),
                        "tau_tail": float(result["tau_tail"]),
                        "backend": str(result["backend"]),
                    }
                )
    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {0.2: color_cycle[1], 0.4: color_cycle[2], 0.8: color_cycle[3]}

    for e in E_VALUES:
        color = colors[e]
        for n0, linestyle in ((0.0, "-"), (1.0, "--")):
            group = df[(df["e"] == e) & (df["n0"] == n0)].sort_values("M_Q")
            ax.plot(
                group["M_Q"],
                group["edot_hat"],
                color=color,
                lw=2.0,
                ls=linestyle,
                label=rf"$e={e:g},\,n_0={int(n0)}$",
            )

    ax.set_xlabel(r"$\mathcal{M}_Q$", fontsize=16, loc="center")
    ax.set_ylabel(r"$\dot e\,(\nu\tilde\Omega/\bar\rho)$", fontsize=16)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_xlim(float(df["M_Q"].min()) * 0.9, float(df["M_Q"].max()) * 1.02)
    ax.legend(fontsize=9.5, loc="best")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.98)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.recompute:
        df = compute_data(args.backend)
        df.to_csv(DATA_PATH, index=False)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError(f"Missing cached data: {DATA_PATH}. Run with --recompute.")

    if not df["converged"].all():
        raise RuntimeError("The cached data include unconverged rows")
    save_plot(df)
    print(f"figure = {OUTPUT_PATH}")
    print(f"data = {DATA_PATH}")


if __name__ == "__main__":
    main()
