"""Paper Fig. 3: quantum-fluid eccentricity scan at nu=0.2.

This preset keeps the quantum eccentricity-scan visual grammar, but uses
``nu=0.2`` and linear y axes. Command-line arguments supplied by the user
override these presets.
"""

from __future__ import annotations

import sys

import plot_quantum_ecc_fluxes as _base


PRESET_ARGS = [
    "--nu",
    "0.2",
    "--linear-y",
    "--output-dir",
    "outputs/paper_fig3_quantum_nu02_ecc_fluxes",
    "--figure-stem",
    "paper_fig3_quantum_nu02_ecc_fluxes",
    "--report-title",
    "Paper Fig. 3: Quantum nu=0.2 Eccentricity Scan",
]


if __name__ == "__main__":
    _base.__doc__ = __doc__
    sys.argv[1:1] = PRESET_ARGS
    _base.main()
