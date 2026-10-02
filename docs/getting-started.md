# Getting Started

## Installation

Install the runtime dependencies in a Python environment:

```powershell
pip install -r requirements.txt
```

CUDA execution requires a CUDA setup supported by Numba. When CUDA is unavailable, use `backend="cpu"` or `backend="auto"`.

## Examples

```powershell
python classical_fluid.py --quantity power --nu 0.25 --e 0.2 --n0 0 --A 0.5 --backend auto
python quantum_fluid.py --quantity power --nu 0.25 --e 0.2 --n0 0 --A 2.0 --backend auto
python edg_single_perturber_coefficients.py --A 0.5 --e 0.2 --jmax 20 --lmax 13
```

```python
from classical_fluid import classical_fluid_power
from edg_single_perturber_coefficients import edg_single_perturber_coefficients
from quantum_fluid import quantum_fluid_power

classical = classical_fluid_power(nu=0.25, e=0.2, n0=0.0, A=0.5, backend="auto")
quantum = quantum_fluid_power(nu=0.25, e=0.2, n0=0.0, A=2.0, backend="auto")
edg = edg_single_perturber_coefficients(A=0.5, e=0.2, jmax=20, lmax=13)

print(classical.value)
print(quantum.value)
print(edg.P_shape)
```
