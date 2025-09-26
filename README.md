# MQC3

**SDK for Measurement-based Quantum Computation with Continuous-variable Cluster states.**

MQC3 is a software development kit for optical quantum computers.  
It helps you design CV measurement‑based programs, connect to the MQC3 cloud, and run those programs on real hardware or simulators.

> Looking for end‑user docs? See the user guide at [docs/source/index.md](docs/source/index.md).

## Requirements

- **Supported Platforms**
  - Linux (Ubuntu 24.04 LTS recommended; WSL2 on Windows 11 is also supported)
  - Windows 11
  - macOS (Apple silicon, ARM64)
- **Python versions**
  - Core library: **3.10, 3.11, 3.12, 3.13**
  - Extra **`[sf]`** (StrawberryFields-based simulator): **3.10–3.12 only**  
    *(🚫 Not supported on Python 3.13)*

> **Why is `[sf]` unavailable on Python 3.13?**  
> The `[sf]` stack requires `scipy < 1.14`. Prebuilt wheels for `scipy < 1.14` are not provided on Python 3.13, which forces a Fortran-based source build.  
> To keep setup simple and reliable, `[sf]` is limited to Python 3.10–3.12.

### Compatibility matrix

|  Extra | 3.10  | 3.11  | 3.12  | 3.13  |
| -----: | :---: | :---: | :---: | :---: |
|   core |   ✓   |   ✓   |   ✓   |   ✓   |
| `[sf]` |   ✓   |   ✓   |   ✓   |   ✗   |

## Installation

Install the core library:

```sh
python -m pip install .
```

By default, only a minimal set of features is installed (e.g., visualization may be unavailable).
To enable all optional features:

```sh
python -m pip install "<path/to/sdk>[all]"
```

To install tools for tests and docs:

```sh
python -m pip install "<path/to/sdk>[dev]"
```

### Installing the StrawberryFields-based simulator ([sf])

[sf] is supported only on Python 3.10–3.12:

```sh
# choose one of 3.10 / 3.11 / 3.12
python -m pip install "<path/to/sdk>[sf]"
```

You can combine extras as needed:

```sh
# simulator + all features (Python 3.10–3.12)
python -m pip install "<path/to/sdk>[sf,all]"

# simulator + dev tools (Python 3.10–3.12)
python -m pip install "<path/to/sdk>[sf,dev]"
```

[sf] is not included in [all]. Combine as [sf,all] when needed (on Python 3.10–3.12).

## Quickstart

After installation, try a minimal program:

```python
from math import pi
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic

# Create a circuit representation of a program
c = CircuitRepr("sample_circuit")
c.Q(0) | intrinsic.PhaseRotation(phi=pi / 2)  # Apply a phase rotation of pi/2 to qumode 0
c.Q(0) | intrinsic.Measurement(theta=0.0)     # Measure qumode 0 (homodyne)

print(c)
```

See the user docs for details: [docs/source/index.md](docs/source/index.md).

## Tests

**Some test suites are long‑running. Use `pytest-xdist` to parallelize.**

```sh
# basic tests
pytest

# tests requiring network access
pytest --network

# tests requiring simulator access
pytest --simulator

# long‑running tests (parallel)
pytest -n auto --longrun

# everything
pytest -n auto --longrun --network --simulator
```

## Project layout

```text
├── docs/
├── src/
│   └── mqc3/
│       ├── circuit/
│       ├── client/
│       ├── execute/
│       ├── feedforward/
│       ├── graph/
│       ├── machinery/
│       └── pb/
└── tests/
```

- `docs/` : User & developer documentation sources.
- `src/mqc3/` : Main source tree.
  - `circuit/` : Circuit representation.
  - `client/` :  Execution client and result types.
  - `execute/` : Unified wrapper over multiple clients; one-call submit & fetch results.
  - `feedforward/` : Mechanisms to update operation parameters conditioned on measurement outcomes.
  - `graph/` : Graph representation.
  - `machinery/` : Machinery representation.
  - `pb/` : Protocol Buffers (auto-generated).
- `tests/` : Unit and integration tests.

## License

MIT License. See [LICENSE](LICENSE).
