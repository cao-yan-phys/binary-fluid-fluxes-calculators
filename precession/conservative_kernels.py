"""Real conservative principal-value response kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


FOUR_PI = 4.0 * math.pi


class ConservativeMedium(Protocol):
    rho_bar: float
    include_self_gravity: bool
    response_prescription: str

    def denominator(self, omega: float, k: float) -> float: ...

    def roots(self, omega: float) -> tuple[float, float | None]: ...

    @property
    def response_strength(self) -> float: ...


def _validate_common(rho_bar: float, response_prescription: str) -> None:
    if rho_bar < 0.0:
        raise ValueError("rho_bar must be non-negative")
    if response_prescription != "time_symmetric":
        raise ValueError(
            "only response_prescription='time_symmetric' is available; "
            "retarded_steady requires a finite stabilized background"
        )


@dataclass(frozen=True)
class ClassicalFluid:
    """Classical-fluid conservative response parameters.

    With self-gravity on, this defines a time-symmetric PV prescription in an
    infinite Jeans-unstable background; it is not a stationary retarded wake.
    """

    rho_bar: float
    c_s: float
    include_self_gravity: bool = True
    response_prescription: str = "time_symmetric"

    def __post_init__(self) -> None:
        _validate_common(self.rho_bar, self.response_prescription)
        if self.c_s <= 0.0:
            raise ValueError("c_s must be positive")

    @property
    def response_strength(self) -> float:
        return FOUR_PI**2 * self.rho_bar

    def a_n(self, omega: float) -> float:
        return omega * omega + (FOUR_PI * self.rho_bar if self.include_self_gravity else 0.0)

    def denominator(self, omega: float, k: float) -> float:
        return self.a_n(omega) - self.c_s * self.c_s * k * k

    def roots(self, omega: float) -> tuple[float, float | None]:
        return math.sqrt(self.a_n(omega)) / self.c_s, None

    def kernel(self, omega: float, k: float) -> float:
        if k == 0.0:
            raise ZeroDivisionError("kernel contains 1/k^2; use radial measure-reduced form")
        return self.response_strength / (k * k * self.denominator(omega, k))


@dataclass(frozen=True)
class QuantumFluid:
    """Physical-time response of a Schrodinger-Poisson quantum fluid."""

    rho_bar: float
    m_phi: float
    c_s: float = 0.0
    include_self_gravity: bool = True
    response_prescription: str = "time_symmetric"
    c_s_squared: float | None = None

    def __post_init__(self) -> None:
        _validate_common(self.rho_bar, self.response_prescription)
        if self.m_phi <= 0.0:
            raise ValueError("m_phi must be positive")
        if self.c_s < 0.0:
            raise ValueError("c_s must be non-negative")
        if self.c_s_squared is not None and not math.isfinite(self.c_s_squared):
            raise ValueError("c_s_squared must be finite")

    @property
    def sound_speed_squared(self) -> float:
        """Return the physical-time coefficient of the ``k^2`` term."""

        return self.c_s * self.c_s if self.c_s_squared is None else self.c_s_squared

    @property
    def response_strength(self) -> float:
        return FOUR_PI**2 * self.rho_bar

    def a_n(self, omega: float) -> float:
        return omega * omega + (FOUR_PI * self.rho_bar if self.include_self_gravity else 0.0)

    def denominator(self, omega: float, k: float) -> float:
        return self.a_n(omega) - self.sound_speed_squared * k * k - k**4 / (4.0 * self.m_phi**2)

    def roots(self, omega: float) -> tuple[float, float | None]:
        a_value = self.a_n(omega)
        c_s_squared = self.sound_speed_squared
        discriminant = math.sqrt(c_s_squared * c_s_squared + a_value / self.m_phi**2)
        k_squared = 2.0 * self.m_phi**2 * (discriminant - c_s_squared)
        kappa_squared = 2.0 * self.m_phi**2 * (discriminant + c_s_squared)
        return math.sqrt(max(0.0, k_squared)), math.sqrt(max(0.0, kappa_squared))

    def kernel(self, omega: float, k: float) -> float:
        if k == 0.0:
            raise ZeroDivisionError("kernel contains 1/k^2; use radial measure-reduced form")
        return self.response_strength / (k * k * self.denominator(omega, k))

    def factorization_residual(self, omega: float, k: float) -> float:
        k_pole, kappa = self.roots(omega)
        assert kappa is not None
        factorized = (k_pole * k_pole - k * k) * (k * k + kappa * kappa) / (4.0 * self.m_phi**2)
        return self.denominator(omega, k) - factorized


def radial_measure_prefactor(medium: ConservativeMedium) -> float:
    """Return `(4*pi)^2 rho_bar/(2*pi)^3` after the `k^2` measure cancels."""

    return medium.response_strength / (2.0 * math.pi) ** 3


def static_ir_is_prescription_dependent(medium: ConservativeMedium) -> bool:
    """Identify the unregulated pure-quantum-pressure static limit."""

    return (
        isinstance(medium, QuantumFluid)
        and not medium.include_self_gravity
        and medium.sound_speed_squared == 0.0
    )
