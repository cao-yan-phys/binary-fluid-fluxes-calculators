"""Conservative periastron precession in homogeneous fluid backgrounds.

The package is independent of the absorptive on-shell flux calculators.  It
uses a time-symmetric principal-value response and physical orbital frequency
``tildeOmega`` throughout its public API.
"""

from .periastron_precession import (
    PrecessionConfig,
    PrecessionResult,
    calculate_precession,
    classical_precession,
    quantum_precession,
)
from .conservative_kernels import ClassicalFluid, QuantumFluid
from .orbit_derivatives import BinaryOrbit
from .parameterization import classical_precession_flux_parameters, quantum_precession_flux_parameters

__all__ = [
    "BinaryOrbit",
    "ClassicalFluid",
    "QuantumFluid",
    "PrecessionConfig",
    "PrecessionResult",
    "calculate_precession",
    "classical_precession",
    "quantum_precession",
    "classical_precession_flux_parameters",
    "quantum_precession_flux_parameters",
]
