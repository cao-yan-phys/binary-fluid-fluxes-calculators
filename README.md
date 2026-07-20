# Binary Fluid Sound-Wave Flux Calculators

Numerical calculators for time-averaged sound-wave fluxes from bound Keplerian binaries in homogeneous Newtonian classical barotropic-fluid and Schrödinger-Poisson quantum-fluid backgrounds.

The code evaluates the harmonic sums with automatic convergence checks and can use CUDA acceleration through `numba.cuda` when available. CPU execution is also supported.

<p align="center"><img src="docs/assets/binary_cm_orbit_angular_power.png" alt="Angular distribution of sound-wave energy flux" width="760"></p>
<p align="center">(parameters: <code>nu=0.20</code>, <code>e=0.45</code>, <code>n0=0</code>, <code>A=a*Omega=0.55</code>)</p>

## Contents

- `classic_fluid_power.py`: classical-fluid normalized power `P/(2*rho_bar*M^2/c_s)`.
- `classic_fluid_tau_z.py`: classical-fluid normalized angular-momentum flux `tau_z*tildeOmega/(2*rho_bar*M^2/c_s)`.
- `classic_fluid_force_y.py`: classical-fluid normalized y-force `F_y/(2*rho_bar*M^2/c_s^2)`.
- `quantum_fluid.py`: quantum-fluid normalized `P`, `tau_z`, and `F_y`.
- `quadrupole_fluxes.py`: general quadrupole-approximation fluxes for classical and quantum fluids.
- `single_perturber_classic.py`: fixed-center single-perturber classical limit.
- `eytan_sound_wave_coefficients.py`: finite-cutoff Eytan--Desjacques--Ginat single-perturber coefficient calculator.
- `classic_fluid_quadrupole.py`: closed-form massless classical quadrupole formulas.
- `examples/quickstart.py`: small smoke-test style usage example.
- `paper_plots/`: one script for each public paper flux figure.

## Parameters

The calculators use dimensionless parameters built from the auxiliary-time wave equation. The main symbols are:

```text
rho_bar     homogeneous background mass density
tildeOmega  physical orbital angular frequency, T = 2*pi/tildeOmega
Omega       orbital frequency in the auxiliary time
m           effective Jeans/tachyonic mass in the auxiliary-time equation
n0          m/Omega
c_s         classical-fluid sound speed
m_phi       scalar-particle mass in the quantum-fluid case
```

The total binary mass is `M = m1 + m2`, and the symmetric mass ratio is

```text
nu = m1*m2/(m1+m2)^2,    0 < nu <= 1/4.
```

The Keplerian orbit uses eccentricity `e` and eccentric anomaly `xi`,

```text
X/a = (cos xi - e, sqrt(1-e^2) sin xi, 0),
tildeOmega*t_phys = xi - e*sin xi.
```

For the classical Newtonian fluid, the code parameter is the Mach number ($\mathcal{M}=a\Omega$):

```text
A = a*Omega.
```

The relation to physical parameters is

```text
t_aux = c_s*t_phys,
Omega = tildeOmega/c_s,
m^2 = 4*pi*rho_bar/c_s^2,
n0 = m/Omega = sqrt(4*pi*rho_bar)/tildeOmega,
rho_bar = (n0*tildeOmega)^2/(4*pi) = c_s^2*m^2/(4*pi).
```

For `n > 0`, the classical radiating wavenumber is

```text
k_n = Omega*sqrt(n^2+n0^2),
a*k_n = A*sqrt(n^2+n0^2).
```

For the Schrödinger-Poisson quantum fluid, the code parameter is the number ($\mathcal{M}_Q=a\sqrt{\Omega}$):

```text
A = a*sqrt(Omega).
```

The relation to physical parameters is

```text
t_aux = t_phys/(2*m_phi),
Omega = 2*m_phi*tildeOmega,
m^2 = 16*pi*m_phi^2*rho_bar,
n0 = m/Omega = sqrt(4*pi*rho_bar)/tildeOmega,
rho_bar = (n0*tildeOmega)^2/(4*pi) = m^2/(16*pi*m_phi^2).
```

For `n > 0`, the radiating wavenumber is

```text
k_n = sqrt(Omega)*(n^2+n0^2)^(1/4),
a*k_n = A*(n^2+n0^2)^(1/4).
```

This is the default $c_S=0$ case. To include a finite quantum-fluid sound term, pass `cS_over_sqrtOmega` or the signed parameter `cS2_over_Omega` to the `quantum_fluid.py` calculators; the corresponding CLI flags are `--cS-over-sqrtOmega` and `--cS2-over-Omega`.

The returned values are dimensionless normalized fluxes. For a homogeneous background density `rho_bar`, the classical-fluid outputs correspond to

```text
P     = (2*rho_bar*M^2/c_s) * P_hat,
tau_z = (2*rho_bar*M^2/c_s) * tau_hat/tildeOmega,
F_y   = (2*rho_bar*M^2/c_s^2) * Fy_hat.
```

For the quantum-fluid outputs,

```text
P_hat   = P/(2*rho_bar*M^2*m_phi/sqrt(Omega)),
tau_hat = tau_z*tildeOmega/(2*rho_bar*M^2*m_phi/sqrt(Omega)),
Fy_hat  = F_y*sqrt(Omega)/m_phi/(2*rho_bar*M^2*m_phi/sqrt(Omega)).
```

Equivalently,

```text
P     = (2*rho_bar*M^2*m_phi/sqrt(Omega)) * P_hat,
tau_z = (2*rho_bar*M^2*m_phi/sqrt(Omega)) * tau_hat/tildeOmega,
F_y   = (2*rho_bar*M^2*m_phi^2/Omega) * Fy_hat.
```

Thus the same dimensionless `n0` is used in both media:

```text
n0 = sqrt(4*pi*rho_bar)/tildeOmega.
```

The direct fixed-center single-perturber calculator
`single_perturber_classic.py` uses `A = a*Omega/c_s` for the perturber orbit and
returns

```text
single_perturber_power().value = P/(2*rho_bar*m_p^2/c_s),
single_perturber_tau_z().value = tau_z*tildeOmega/(2*rho_bar*m_p^2/c_s).
```

Here `m_p` is the perturber mass.

The Eytan--Desjacques--Ginat helper `eytan_sound_wave_coefficients.py` is a
fixed-center single-perturber coefficient calculator.  Its `A` argument is
`a*Omega/c_s` for the single-perturber orbit.  The returned `P_shape` and
`tau_z_shape` are converted to the single-perturber normalized fluxes by

```text
P/(2*rho_bar*m_p^2/c_s) = 2*pi*P_shape,
tau_z*tildeOmega/(2*rho_bar*m_p^2/c_s) = 2*pi*A*tau_z_shape.
```

## Installation

Create a Python environment and install the runtime dependencies:

```powershell
pip install -r requirements.txt
```

CUDA acceleration requires a working NVIDIA CUDA setup supported by Numba. If CUDA is unavailable, use `backend="cpu"` or `backend="auto"`.

## Quick Start

```powershell
python examples/quickstart.py
```

Direct CLI example:

```powershell
python classic_fluid_power.py --nu 0.25 --e 0.2 --n0 0 --A 0.5 --backend auto
python quantum_fluid.py --quantity power --nu 0.25 --e 0.2 --n0 0 --A 2.0 --backend auto
python eytan_sound_wave_coefficients.py --A 0.5 --e 0.2 --jmax 20 --lmax 13
```

Minimal Python example:

```python
from classic_fluid_power import classical_fluid_power
from eytan_sound_wave_coefficients import eytan_sound_wave_coefficients
from quantum_fluid import quantum_fluid_power

classic = classical_fluid_power(nu=0.25, e=0.2, n0=0.0, A=0.5, backend="auto")
quantum = quantum_fluid_power(nu=0.25, e=0.2, n0=0.0, A=2.0, backend="auto")
eytan = eytan_sound_wave_coefficients(A=0.5, e=0.2, jmax=20, lmax=13)

print(classic.value)
print(quantum.value)
print(eytan.P_shape)
```

## Paper Plots

The scripts in `paper_plots/` write outputs below `outputs/paper_plots/`.

```powershell
python paper_plots/plot_circular_power_nu.py
python paper_plots/plot_paper_fig1_emri_fluxes.py
python paper_plots/plot_paper_fig3_nu02_ecc_fluxes.py
python paper_plots/plot_quantum_equal_mass_power_curves.py
python paper_plots/plot_paper_fig3_quantum_nu02_ecc_fluxes.py
```

## Notes

- The infinite harmonic sum is not returned at an arbitrary cutoff by default. The calculators evaluate chunks of harmonics until recent tail contributions are below the requested tolerance for several consecutive checks.
- Classical-fluid point-source calculations have a built-in large-harmonic speed-threshold guard.
- Please verify convergence settings for each new parameter regime.
- The quantum fluid is described classically.

## References

- G. Eytan, V. Desjacques, and Y. B. Ginat, [arXiv:2509.15632](https://arxiv.org/abs/2509.15632). We only implement the single-perturber version in `eytan_sound_wave_coefficients.py`.
