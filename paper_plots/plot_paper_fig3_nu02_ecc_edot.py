from __future__ import annotations

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


DATA_PATH = Path("outputs/paper_plots/paper_fig3_nu02_ecc_fluxes_data.csv")
OUTPUT_PATH = Path("outputs/paper_plots/paper_fig3_nu02_ecc_edot.pdf")
E_VALUES = (0.2, 0.4, 0.8)


def add_edot_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df[df["e"].isin(E_VALUES)].copy()
    eccentricity = result["e"].to_numpy(dtype=float)
    result["edot_hat"] = -2.0 * result["Mach"] * (1.0 - eccentricity * eccentricity) / eccentricity * (
        result["P_hat"] - result["tau_hat"] / np.sqrt(1.0 - eccentricity * eccentricity)
    )
    return result


def save_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {0.2: color_cycle[1], 0.4: color_cycle[2], 0.8: color_cycle[3]}

    for e in E_VALUES:
        group_e = df[df["e"] == e]
        color = colors[e]
        for n0, linestyle in ((0.0, "-"), (1.0, "--")):
            group = group_e[group_e["n0"] == n0].sort_values("Mach")
            ax.plot(
                group["Mach"],
                group["edot_hat"],
                color=color,
                lw=2.0,
                ls=linestyle,
                label=rf"$e={e:g},\,n_0={int(n0)}$",
            )
        ax.axvline(group_e["Mcrit"].iloc[0], color=color, lw=1.2, ls=":", alpha=0.75)

    ax.set_xlabel(r"$\mathcal{M}$", fontsize=16, loc="center")
    ax.set_ylabel(r"$\dot e\,(\nu\tilde\Omega/\bar\rho)$", fontsize=16)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_xlim(float(df["Mach"].min()) * 0.9, float(df["Mcrit"].max()) * 1.05)
    ax.legend(fontsize=9.5, loc="lower left")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.98)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing cached flux data: {DATA_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = add_edot_column(pd.read_csv(DATA_PATH))
    save_plot(df)
    print(f"figure = {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
