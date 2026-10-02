# Calculators

- `classical_fluid.py`: normalized classical-fluid energy, angular-momentum, and linear-momentum fluxes, $P$, $\tau_z$, and $F_y$.
- `quantum_fluid.py`: normalized quantum-fluid energy, angular-momentum, and linear-momentum fluxes, $P$, $\tau_z$, and $F_y$.
- `quadrupole_fluxes.py`: classical-fluid and $c_S=0$ quantum-fluid fluxes at quadrupole order (quadrupole + monopole radiation).
- `classical_fluid_quadrupole.py`: closed-form $n_0=0$ classical-fluid fluxes at quadrupole order.
- `single_perturber_classical.py`: classical-fluid flux calculator for a single perturber, using the same harmonic-sum method as the binary calculators.
- `edg_single_perturber_coefficients.py`: finite-cutoff calculator for the $n_0=0$ Eytan–Desjacques–Ginat single-perturber coefficients.
- `examples/quickstart.py`: minimal usage example and smoke test.

## Single Perturber

`single_perturber_classical.py` uses the same classical parameter,

$$
A=a\Omega=\frac{a\tilde\Omega}{c_s},
$$

and returns

```text
single_perturber_power().value = P/(2*rho_bar*m_p^2/c_s)
single_perturber_tau_z().value = tau_z*tildeOmega/(2*rho_bar*m_p^2/c_s)
single_perturber_force_y().value = F_y/(2*rho_bar*m_p^2/c_s^2)
```

Here, $m_p$ denotes the perturber mass.

## Eytan–Desjacques–Ginat Coefficients

The `edg_single_perturber_coefficients.py` helper implements the $n_0=0$ single-perturber coefficients of Eytan, Desjacques, and Ginat ([arXiv:2509.15632](https://arxiv.org/abs/2509.15632)). It uses $A=a\Omega=a\tilde\Omega/c_s$. The returned `P_shape` and `tau_z_shape` are converted to normalized single-perturber fluxes according to

$$
\frac{P}{2\bar\rho m_p^2/c_s}=2\pi\,P_{\rm shape},
\qquad
\frac{\tau_z\tilde\Omega}{2\bar\rho m_p^2/c_s}=2\pi A\,\tau_{z,{\rm shape}}.
$$
