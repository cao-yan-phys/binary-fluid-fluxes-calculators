# Classical-fluid normalized power

This note transcribes the formulas needed from the problem PDF for the
classical-fluid calculation.

## Orbit and angular direction

For a Keplerian ellipse,

```text
X1 = (m2/M) X,    X2 = -(m1/M) X,    M = m1 + m2
```

with eccentric anomaly `xi`,

```text
X/a = (cos(xi) - e, sqrt(1 - e^2) sin(xi), 0)
Omega t = xi - e sin(xi)
```

and

```text
n_hat = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)).
```

Therefore

```text
n_hat . (X/a)
  = sin(theta) [cos(phi) (cos(xi) - e)
      + sin(phi) sqrt(1 - e^2) sin(xi)].
```

## Mode amplitude

For harmonic `n >= 1`,

```text
K_n(theta, phi) =
  int_0^(2pi) dxi/(2pi) (1 - e cos(xi)) exp(i n (xi - e sin(xi)))
  * [
      (m1/M) exp(-i (m2/M) (a k_n) n_hat . (X/a))
      + (m2/M) exp( i (m1/M) (a k_n) n_hat . (X/a))
    ].
```

The implementation uses the dimensionless symmetric mass ratio

```text
nu = m1 m2 / M^2,    0 < nu <= 1/4
```

and reconstructs

```text
m1/M = (1 + sqrt(1 - 4 nu)) / 2,
m2/M = (1 - sqrt(1 - 4 nu)) / 2.
```

Swapping labels only changes the phase convention; the energy flux depends on
`|K_n|^2`.

## Classical-fluid dispersion and target normalization

From the classical-fluid page,

```text
k_n = Omega_n sqrt(1 + (m/Omega_n)^2),
Omega_n = n Omega,
n0 = m/Omega.
```

With the requested parameter

```text
A = a Omega,
```

the dimensionless phase factor is

```text
a k_n = A n sqrt(1 + (n0/n)^2).
```

The requested calculator returns

```text
P / (2 bar(rho) M^2 / c_s)
  = sum_{n>=1} [ int sin(theta) dtheta dphi |K_n|^2 ]
      / sqrt(1 + (n0/n)^2).
```

The z-component torque calculator returns the similarly normalized quantity

```text
tau_z * tilde(Omega) / (2 bar(rho) M^2 / c_s)
  = Re sum_{n>=1} [(-i)/n] / sqrt(1 + (n0/n)^2)
      * int sin(theta) dtheta dphi [-K_n partial_phi K_n^*].
```

Writing `K_n = a + i b` and `partial_phi K_n = c + i d`, the real contribution
used numerically is

```text
[a d - b c] / [n sqrt(1 + (n0/n)^2)].
```

The y-component force calculator returns

```text
F_y / (2 bar(rho) M^2 / c_s^2)
  = sum_{n>=1} int sin(theta) dtheta dphi |K_n|^2 sin(theta) sin(phi).
```

Unlike the power expression, this Eq. (2.52) force expression carries no
additional `1/sqrt(1 + (n0/n)^2)` weight.

The angular integral is evaluated as

```text
int sin(theta) dtheta dphi = int_{-1}^{1} dmu int_0^{2pi} dphi,
mu = cos(theta).
```

The code uses Gauss-Legendre quadrature in `mu`, a uniform trapezoidal rule in
`phi`, and a uniform periodic trapezoidal rule in `xi`.

## Automatic harmonic-sum convergence

The infinite harmonic sum is not returned at a user-chosen truncation by
default.  The code evaluates harmonics in chunks until

```text
max(sum(last tail_window terms), sum(current chunk terms))
  <= atol + rtol * abs(cumulative sum)
```

for several consecutive checks.  `n_max` is only a safety cap.  If the safety
cap is reached first, the default behavior is to raise an error instead of
returning a truncated value.

When `n_xi` is not fixed by the user, the `xi` quadrature is also increased
with the current largest harmonic:

```text
n_xi = max(512, xi_per_n * current_chunk_stop).
```

## Large-n speed threshold

The phase in `K_n` contains the barycentric source positions, not the relative
separation alone:

```text
X1 = (m2/M) X,    X2 = -(m1/M) X.
```

Thus a stationary large-`n` phase first appears when one body, not the relative
coordinate, crosses the sound-speed threshold:

```text
q_max * A * sqrt((1 + e)/(1 - e)) >= 1,
q_max = max(m1/M, m2/M).
```

Equivalently,

```text
A >= 1 / [q_max * sqrt((1 + e)/(1 - e))].
```

The often-quoted relative-orbit condition is recovered only by setting
`q_max = 1`.  For equal masses, `q_max = 1/2`, so the threshold is twice the
relative-orbit value.

The calculator uses this as a default guard: parameters with threshold ratio
`>= 1` raise a divergence error.  Diagnostic plotting scripts disable that guard
explicitly so finite-cutoff partial sums can still be inspected.

## Classical n0=0 quadrupole approximation

For the massless classical fluid (`n0 = 0`), the small-`A` quadrupole
approximation is

```text
P = (4 pi / 15) * nu^2 M^2 rho_bar Omega^4 a^4 / c_s
    * [7 sqrt(1-e^2) - 3(1-e^2)] / (1-e^2),

tau_z = (16 pi / 15) * nu^2 M^2 rho_bar Omega^3 a^4 / c_s^2
    * sqrt(1-e^2).
```

In the numerical calculator's dimensionless variables,

```text
A = a tilde(Omega) / c_s,
```

the corresponding normalized quantities are

```text
P / (2 rho_bar M^2 / c_s)
  = (2 pi / 15) * nu^2 * A^4
    * [7 sqrt(1-e^2) - 3(1-e^2)] / (1-e^2),

tau_z * tilde(Omega) / (2 rho_bar M^2 / c_s)
  = (8 pi / 15) * nu^2 * A^4 * sqrt(1-e^2).
```

These are implemented in `classic_fluid_quadrupole.py`; the comparison script is
`check_classic_quadrupole.py`.

# Quantum-fluid normalized observables

For the quantum-fluid page,

```text
k_n = (Omega_n^2 + m^2)^(1/4),
Omega_n = n Omega,
n0 = m/Omega.
```

The phase factor in `K_n` is therefore

```text
a k_n = A (n^2 + n0^2)^(1/4),    A = a sqrt(Omega).
```

The same `K_n(theta, phi)` definition is used, with this quantum value of
`a k_n`.

The quantum power calculator returns

```text
P / (2 bar(rho) M^2 m_phi / sqrt(Omega))
  = sum_{n>=1} n / (n^2 + n0^2)^(3/4)
      * int sin(theta) dtheta dphi |K_n|^2.
```

The quantum z-torque calculator returns

```text
tau_z tilde(Omega) / (2 bar(rho) M^2 m_phi / sqrt(Omega))
  = Re sum_{n>=1} (-i) / (n^2 + n0^2)^(3/4)
      * int sin(theta) dtheta dphi [-K_n partial_phi K_n^*].
```

The quantum force expression on the PDF is along `y`.  The calculator returns

```text
[F_y sqrt(Omega)/m_phi] / [2 bar(rho) M^2 m_phi / sqrt(Omega)]
  = sum_{n>=1} 2 / sqrt(n^2 + n0^2)
      * int sin(theta) dtheta dphi |K_n|^2 sin(theta) sin(phi).
```
