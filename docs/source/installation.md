---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 4.0.0
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Installation

MQC3 has been tested in the following environments:

- **Python**: 3.10, 3.11, 3.12, 3.13  
- **OS**: Ubuntu 24.04

Before proceeding, ensure your Python version is **3.10 or higher**:

```sh
python3 --version
```

## Quick install

Replace `<path/to/sdk>` with the local path to the MQC3 directory:

```sh
python3 -m pip install <path/to/sdk>
```

````{note}
By default, only a minimal set of features is installed (e.g., visualization may be unavailable).  
To enable optional features, use extras:

- **All optional features**
  ```sh
  python3 -m pip install <path/to/sdk>[all]
  ```

- **Development tools** (tests, docs, linters, etc.)
  ```sh
  python3 -m pip install <path/to/sdk>[dev]
  ```

- **StrawberryFields-based simulator** (`[sf]`) — **Python 3.10–3.12 only**
  ```sh
  # choose one of 3.10 / 3.11 / 3.12
  python3.12 -m pip install <path/to/sdk>[sf]
  ```

You can combine extras as needed:

- Simulator + all features (Python 3.10–3.12)
  ```sh
  python3.12 -m pip install <path/to/sdk>[sf,all]
  ```

- Simulator + development tools (Python 3.10–3.12)
  ```sh
  python3.12 -m pip install <path/to/sdk>[sf,dev]
  ```
````

````{warning}
The **`[sf]`** extra is **not supported on Python 3.13**.  
It relies on a dependency stack that requires `scipy < 1.14`, for which prebuilt wheels are not available on Python 3.13.  
Use Python **3.10–3.12** when installing `[sf]`.
````

**Compatibility matrix**

| Extra | 3.10 | 3.11 | 3.12 | 3.13 |
|------:|:----:|:----:|:----:|:----:|
| core  |  ✓   |  ✓   |  ✓   |  ✓   |
| `[sf]`|  ✓   |  ✓   |  ✓   |  ✗   |

## Verify installation

```{code-cell} python
>>> import mqc3
>>> mqc3.__version__
```
