"""Small usage example for the binary fluid calculators."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic_fluid_force_y import classical_fluid_force_y
from classic_fluid_power import classical_fluid_power
from classic_fluid_tau_z import classical_fluid_tau_z
from eytan_sound_wave_coefficients import eytan_sound_wave_coefficients
from quadrupole_fluxes import quantum_quadrupole_flux_normalized
from quantum_fluid import quantum_fluid_force_y, quantum_fluid_power, quantum_fluid_tau_z


def compact_kwargs(**overrides):
    kwargs = dict(
        nu=0.25,
        e=0.2,
        n0=0.0,
        n_max=512,
        n_mu=16,
        n_phi=32,
        backend="auto",
        chunk_size=64,
        rtol=1.0e-5,
        tail_window=16,
        consecutive_windows=2,
        strict_convergence=False,
        xi_per_n=4,
    )
    kwargs.update(overrides)
    return kwargs


def main() -> None:
    classic_common = compact_kwargs(A=0.5)
    p_classic = classical_fluid_power(**classic_common)
    tau_classic = classical_fluid_tau_z(**classic_common)
    fy_classic = classical_fluid_force_y(**classic_common)

    quantum_common = compact_kwargs(A=2.0)
    p_quantum = quantum_fluid_power(**quantum_common)
    tau_quantum = quantum_fluid_tau_z(**quantum_common)
    fy_quantum = quantum_fluid_force_y(**quantum_common)

    quad_quantum = quantum_quadrupole_flux_normalized(
        nu=0.25,
        e=0.2,
        n0=0.0,
        A=0.2,
        n_max=256,
        strict_convergence=False,
    )
    eytan = eytan_sound_wave_coefficients(
        A=0.5,
        e=0.2,
        jmax=8,
        lmax=8,
        n_xi=2048,
    )

    print("Classical normalized fluxes")
    print(f"  P_hat       = {p_classic.value:.8e}")
    print(f"  tau_hat     = {tau_classic.value:.8e}")
    print(f"  F_y_hat     = {fy_classic.value:.8e}")
    print(f"  converged   = {p_classic.converged}, n={int(p_classic.n_values[-1])}")

    print("\nQuantum normalized fluxes")
    print(f"  P_hat       = {p_quantum.value:.8e}")
    print(f"  tau_hat     = {tau_quantum.value:.8e}")
    print(f"  F_y_hat     = {fy_quantum.value:.8e}")
    print(f"  converged   = {p_quantum.converged}, n={int(p_quantum.n_values[-1])}")

    print("\nQuantum quadrupole approximation")
    print(f"  P_hat       = {quad_quantum.P:.8e}")
    print(f"  tau_hat     = {quad_quantum.tau_z_tildeOmega:.8e}")

    print("\nEytan single-perturber coefficient check")
    print(f"  IE/A        = {eytan.P_shape:.8e}")
    print(f"  IL/A^2      = {eytan.tau_z_shape:.8e}")


if __name__ == "__main__":
    main()
